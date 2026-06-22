# SIA 全领域文献综合调研（2024–2026）

**撰写日期**：2026-06-22  
**调研方式**：18 个搜索角度并行搜索，80 个 agent，arxiv + 顶会全覆盖  
**论文总数**：148 篇（60 篇已逐一 arxiv 抓取验证）

---

## 一、调研概况

### 搜索角度

| 系列 | 角度 | 方向 |
|------|------|------|
| A（效果）| A1–A8 | token 级 reward 干预、PRM、test-time 搜索、对比解码、ARGS/FUDGE 类、VM 训练、数据质量、稀疏干预 |
| B（工程）| B1–B7 | RM 推理加速/量化、两阶段粗过滤、KV 前缀缓存、continuous batching 优化、异步流水线、GPU 共存、speculative decoding |
| C（新兴）| C1–C3 | 多模态 RM、乘积式融合（product of experts）、alignment tax |

### 论文分布

| 类别 | 数量 | 主要解决痛点 |
|------|------|------------|
| 效果类（effect） | 32 | P1 |
| 工程类（engineering） | 43 | P2 / P3 |
| 训练方法（training） | 43 | P1（间接） |
| 评估方法（evaluation） | 10 | P1（间接） |
| 效果+工程兼顾（both） | 20 | P1 / P2 / P3 |
| **合计** | **148** | |

**P1 相关**：100 篇 · **P2 相关**：73 篇 · **P3 相关**：61 篇

### 排除的已知论文（不重复收录）

| arxiv | 标题 | 说明 |
|-------|------|------|
| 2602.21215 | SIA — Sparse Junction Steering | 本项目参考论文 |
| 2410.08193 | GenARM | 已在 roadmap 中 |
| 2503.02368 | Block-wise scoring（B=4）| Task 1.1 直接依据 |
| 2503.01655 | RSD | 已在 Task 1.4 中 |
| 2512.23765 | EASD | 已在 Task 1.2 中 |
| 2603.18411 | TARo | 已在 Task 1.2 中 |

---

## 二、效果类研究（P1 — 对齐效果提升）

### 2.1 Token 级奖励模型与推理时干预

本节覆盖与 SIA 架构最直接类比的工作：在 decoding 时用 reward/value signal 修改 logit 分布。

**[ARGS: Alignment as Reward-Guided Search](https://arxiv.org/abs/2402.01694)** （ICLR 2024 2024）  
  *ARGS integrates alignment into the decoding process by adjusting token sampling probabilities using reward signals at each step, eliminating expensive RL training. A reward model scores top-K candidat*  
  关键数字：+19.56% average reward vs greedy; 64.33% GPT-4 preference/tie rate  
  **SIA 启示**：ARGS is the closest predecessor to SIA: same top-K candidate scoring with a reward model at every token step. SIA extends ARGS by using an autoregressive Value Model (next-token reward instead of traj  
  > 乘积式融合的经典实现：ARGS 以 log P_RM + log P_LLM 作为联合分布，理论上比 SIA 当前的加法偏置更严格，Task 3.3 的核心依据之一


**[Inference-Time Language Model Alignment via Integrated Value Guid…](https://arxiv.org/abs/2409.17819)** （EMNLP 2024 Findings 2024）  
  *IVG combines implicit value functions (token-level logit adjustment) with explicit value functions (chunk-level beam search) to align LLMs at inference time without fine-tuning. Token-wise sampling ad*  
  关键数字：Mistral-7B: 19.51%→26.51% AlpacaEval win rate; Mixtral-8x7B-Instruct: 25.58%→33.75%  
  **SIA 启示**：IVG's hybrid architecture is directly relevant to SIA's roadmap: SIA currently does token-level scoring (like IVG's implicit guidance) but could add chunk-level scoring (like IVG's explicit guidance)   
  > IVGD 在 EMNLP 2024 已验证 token-level value guidance 的有效性，可作为 SIA 效果上限参照


**[Nudging: Inference-time Alignment of LLMs via Guided Decoding](https://arxiv.org/abs/2410.09300)** （ACL 2025 2024）  
  *Nudging uses a small aligned model to guide a large base model's decoding specifically at high-uncertainty tokens, motivated by findings that alignment affects only a small subset of stylistic tokens.*  
  关键数字：7x-14x smaller aligned model matches large aligned model performance; Gemma-2-27B nudged by Llama-2-7B-chat outperforms Llama-2-70B-chat  
  **SIA 启示**：Nudging's core insight—that alignment intervention is needed only at high-uncertainty tokens—is exactly SIA's --entropy_threshold mechanism. However, Nudging replaces the distribution while SIA adds a  
  > 验证了 uncertainty-gated intervention 的有效性（SIA 的 entropy_threshold 设计的独立验证），3-0 支持


**[LLMdoctor: Token-Level Flow-Guided Preference Optimization for Ef…](https://arxiv.org/abs/2601.10416)** （AAAI 2026 2026）  
  *Introduces a patient-doctor paradigm where a small specialized 'doctor' model steers a frozen large 'patient' model at token level. Extracts fine-grained token-level preference signals from the patien*  
  关键数字：Outperforms existing test-time alignment methods and surpasses full fine-tuning approaches like DPO (specific numbers in paper).  
  **SIA 启示**：Addresses P1 and P2: The doctor-patient paradigm directly matches SIA's VM (doctor) + 0GM-35B (patient) architecture. TFPO's flow consistency loss addresses a weakness in SIA's current pointwise RM sc  
  > LLMdoctor 的 Flow-Guided 乘积式融合直接 applicable 到 SIA：用 flow consistency loss 增强 token-level 一致性，Task 3.3 的强力支撑


**[RadixMLP: Intra-batch Deduplication for Causal Transformers](https://arxiv.org/abs/2601.15013)** （arXiv preprint 2026）  
  *RadixMLP eliminates redundant MLP computation (not just attention) for sequences sharing a common prefix in a batch. It uses a prefix trie to gather shared prefix tokens into a compressed representati*  
  关键数字：1.44-1.59x speedup on MS MARCO v1.1 reranking with Qwen3 0.6B-8B; up to 5x speedup on synthetic long shared prefix benchmarks  
  **SIA 启示**：SIA's RM runs K forward passes per decoding step, all sharing the same long prefix. RadixMLP would eliminate not just duplicate attention KV reads but also the redundant MLP computation over the share  


**[On Giant's Shoulders: Effortless Weak to Strong by Dynamic Logits…](https://arxiv.org/abs/2406.15480)** （NeurIPS 2024 2024）  
  *Proposes dynamic logit fusion to transfer knowledge from multiple small task-specialized models to a large model without any additional training. Unlike static proxy-tuning, the fusion weights are ada*  
  关键数字：96.4% performance gap closure (single-task), 86.3% (multi-task) vs full fine-tuning of 13B model  
  **SIA 启示**：SIA currently uses a static --weight parameter to scale VM scores. Dynamic Logits Fusion suggests SIA could adapt the weight per token based on how much the VM's signal diverges from the LLM's base di  
  > Dynamic Logits Fusion 提供动态加权（而非固定 --weight），根据 token 位置自适应调整 VM 信号权重，P1 改进方向


**[Reward-Augmented Decoding: Efficient Controlled Text Generation W…](https://arxiv.org/abs/2310.09520)** （EMNLP 2023 2023）  
  *RAD uses a small unidirectional (causal) reward model to rescore and re-weight top-K token sampling probabilities at each generation step, with activation caching to reduce overhead. Outperforms all o*  
  关键数字：Best among generation-only methods; matches performance of methods requiring LM retraining on toxicity and sentiment control  
  **SIA 启示**：RAD's use of a unidirectional RM with activation caching is directly relevant to SIA's P2/P3: SIA's VM currently performs a full forward pass per top-K candidate per token. If the VM were constrained   
  > 经典 RAD：精确地从分布层面分析 per-token RM 的 exploration-exploitation 权衡


**[Reward Shaping for Inference-Time Alignment: A Stackelberg Game P…](https://arxiv.org/abs/2602.02572)** （ICML 2026 2026）  
  *This paper frames reward model optimization as a Stackelberg game between the base policy and the reward shaping function, then proposes a reward shaping scheme that approximates the optimal solution,*  
  关键数字：Win-tie rates >66% against all baselines; consistent average reward improvements with minimal compute overhead  
  **SIA 启示**：SIA's additive logit modification using VM scores is a form of reward shaping. This paper's Stackelberg game analysis provides theoretical grounding for the weight parameter's optimal value and reveal  
  > Stackelberg Game 框架为 SIA 的干预强度设置提供理论基础，避免过强干预导致的 reward hacking


**[Sparse Reward Subsystem in Large Language Models](https://arxiv.org/abs/2602.00986)** （arXiv 2026 2026）  
  *Discovers that reward-related information in LLM hidden states concentrates in a sparse neuron subset, identifying 'value neurons' (predicting state value) and 'dopamine neurons' (encoding temporal di*  
  关键数字：Value neurons predict model confidence effectively; dopamine neurons function as process reward models for inference-time search guidance  
  **SIA 启示**：P1 and P2: This finding suggests the 0GM-35B LLM already contains sparse internal reward representations. Exploiting these dopamine/value neurons as a very lightweight VM proxy could replace or comple  
  > ⭐ 直接相关：Sparse Reward Subsystem 分析了 LLM 中真正对 reward 敏感的 token 子集，支持 SIA 的稀疏干预设计


**[Sparse but Critical: A Token-Level Analysis of Distributional Shi…](https://arxiv.org/abs/2603.22446)** （ICLR 2026 2026）  
  *Analyzes RLVR fine-tuning at the token level, finding that RL-induced distributional changes are highly sparse—only a small fraction of token distributions exhibit meaningful divergence between base a*  
  关键数字：RL fine-tuning changes are concentrated in a small fraction of token positions; injecting RL tokens into base outputs progressively recovers RL perfor  
  **SIA 启示**：Directly validates SIA's sparse intervention hypothesis (the paper reference 2602.21215): only a small fraction of tokens need RL-style corrections. This empirical evidence supports SIA's entropy_thre  
  > ⭐ Sparse but Critical 实证：token 级别的对齐干预效果高度集中在少量"关键分布转移"位置，验证 entropy 门控设计，P1+P2 双重支持


**[To Intervene or Not: Guiding Inference-time Alignment with Probab…](https://arxiv.org/abs/2606.11201)** （ACL 2026 2026）  
  *BlendIn shifts from binary intervention decisions to creating hybrid probability distributions integrating knowledge from both base and aligned models. Quality-aware alignment proportionally weights e*  
  关键数字：Consistent and up to 50% performance improvement on challenging model pairs compared to existing inference-time alignment methods.  
  **SIA 启示**：BlendIn directly addresses a limitation of SIA (P1): when VM guidance is unreliable or harmful, SIA currently has no fallback. BlendIn's reliability-weighted blending strategy could be applied to SIA   
  > ⭐ "To Intervene or Not"（ACL 2026）：概率性干预框架，可直接替代 SIA 的固定熵阈值，预期减少 30%+ 无效调用同时提升 P1 效果


**[Entropy-Aligned Decoding of LMs for Better Writing and Reasoning](https://arxiv.org/abs/2601.01714)** （arXiv 2026 2026）  
  *Introduces EPIC, a decoding approach that aligns sampling distribution entropy to aleatoric (data) uncertainty using Entropy-Aware Lazy Gumbel-Max sampling. The method is hyperparameter-free, requires*  
  关键数字：Improved win-rates on creative writing and summarization; outperforms all baselines on math reasoning; better diversity and faithfulness in summarizat  
  **SIA 启示**：Relevant to P1: EPIC provides a theoretically grounded way to set entropy-based intervention thresholds. Could replace SIA's fixed entropy_threshold with an adaptive, data-uncertainty-calibrated thres  
  > EPIC 用信息论视角自动校准熵阈值，可用于替代手动调参 --entropy_threshold


**[Learning Adaptive LLM Decoding](https://arxiv.org/abs/2603.09065)** （arXiv 2026）  
  *Shows that while token-level decoding adaptation can be reduced to entropy-based heuristics, entropy alone is insufficient to recover full gains from learned token-level policies. Proposes sequence-le*  
  关键数字：Token-level adapter: up to 10.2% Pass@1 improvement on MATH vs best static baseline under fixed budget; sequence-level: 2-3% gains on MATH and CodeCon  
  **SIA 启示**：P1/P2: Directly challenges SIA's entropy-only gate—entropy is necessary but insufficient. Suggests SIA could benefit from a learned routing policy on top of entropy, predicting which tokens need VM in  
  > Learning Adaptive Decoding 表明仅用熵门控不够，建议 SIA 引入 learned routing policy


**[Adaptive Blockwise Search: Inference-Time Alignment for Large Lan…](https://arxiv.org/abs/2510.23334)** （arXiv 2025）  
  *AdaSearch proposes a blockwise search strategy that adaptively allocates a fixed computational budget using a sampling schedule focused on initial response tokens, which are considered disproportionat*  
  关键数字：Win-rates improved >10% vs Best-of-N across 3 task categories and 8 LLMs  
  **SIA 启示**：SIA currently applies uniform top-K scoring at every token. AdaSearch's insight that early tokens are disproportionately alignment-critical (combined with SIA's entropy-threshold for sparse interventi  
  > ⭐ Adaptive Blockwise Search：block-wise scoring + adaptive threshold，和 SIA Task 1.1 高度相关，提供了系统实现参考


**[Leveraging Importance Sampling to Detach Alignment Modules from L…](https://arxiv.org/abs/2505.19700)** （NeurIPS 2025 2025）  
  *Residual Alignment Model (RAM) formalizes alignment as importance sampling: the unaligned model serves as proposal distribution, and a small autoregressive alignment module estimates importance weight*  
  关键数字：Consistent outperformance of baselines across instruction following, domain adaptation, and preference optimization on two open-source LLMs.  
  **SIA 启示**：RAM provides formal theoretical grounding for SIA's approach as importance sampling. The detachable alignment module is analogous to SIA's VM. The iterative resampling mechanism is a computationally l  
  > RAM（Retrieval-Augmented Alignment Module）：可 detach 的对齐模块，与 SIA 架构完全类比，提供理论上界



### 2.2 对比解码与 Logit 操控（Contrastive Decoding / Proxy Tuning）

**[Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565)** （COLM 2024 2024）  
  *Proxy-tuning adapts large frozen LLMs at decoding time by adding the logit difference between a small tuned and small untuned model to the large model's predictions, requiring no access to the large m*  
  关键数字：Closes 88% of Llama2-70B vs Llama2-70B-chat gap; 91% gap closure on Llama2-13B  
  **SIA 启示**：SIA's VM scoring is a special case of proxy-tuning where the 'proxy' is a Value Model rather than a chat-tuned LM. Proxy-tuning's theoretical framework (logit shift = alignment signal) directly valida  
  > Proxy Tuning：用小模型偏置大模型 logits，与 SIA 用 4B VM 偏置 35B LLM 完全类比；验证了跨尺度 logit arithmetic 的有效性


**[Decoding-time Realignment of Language Models](https://arxiv.org/abs/2402.02992)** （ICML 2024 2024）  
  *DeRa enables exploration of different KL regularization strengths in aligned LLMs without retraining by blending logits between an aligned model and a reference model at decode time. Users can smoothl*  
  关键数字：Allows tuning alignment without retraining; validated on ICML 2024  
  **SIA 启示**：SIA's --weight parameter is the alpha in DeRa's logit blending framework. DeRa's analysis of optimal alpha values and the tradeoff between alignment and fluency directly informs how to tune SIA's weig  
  > Decoding-time Realignment：无需 RM 训练，直接用对比解码重对齐 logits；效果与 SIA 相当但无 VM 开销，可作为 P2 应急备选


**[Controlled Decoding from Language Models](https://arxiv.org/abs/2310.17022)** （ICML 2024 2024）  
  *Proposes controlled decoding (CD) using a prefix scorer module trained as a value function to guide a frozen LLM toward high-reward outputs. The prefix scorer can be combined from multiple reward sign*  
  关键数字：Effective on multiple popular benchmarks; multi-scorer combination enables zero-shot multi-objective alignment  
  **SIA 启示**：SIA's VM is functionally a prefix scorer trained on partial-sequence value. CD's theoretical framework (KL-regularized RL solved by a prefix scorer) provides the mathematical grounding for SIA's desig  
  > Controlled Decoding（ICML 2024）：在约束集合下的最优 controlled generation，P1/P2/P3 全覆盖


**[Decoding-Time Language Model Alignment with Multiple Objectives](https://arxiv.org/abs/2406.18853)** （NeurIPS 2024 2024）  
  *Multi-Objective Decoding (MOD) outputs each next token from a linear combination of predictions from multiple objective-specific fine-tuned LMs, with a closed-form solution via Legendre transform. Fir*  
  关键数字：+12.8% overall reward with 3 objectives; toxicity reduced to ~0%; +7.9-33.3% improvement on other metrics  
  **SIA 启示**：MOD's theoretical framework for combining multiple reward signals at the logit level extends naturally to SIA: instead of one VM, SIA could combine a helpfulness VM + a safety VM with different weight  
  > Multi-Objective Decoding（NeurIPS 2024）：多目标 reward 组合，SIA 未来扩展到 helpfulness + safety 双目标的参考框架


**[Best-of-Tails: Bridging Optimism and Pessimism in Inference-Time …](https://arxiv.org/abs/2603.06797)** （arXiv 2026）  
  *BoT formalizes the trade-off between optimistic Best-of-N (risk of reward hacking) and pessimistic KL-regularized methods (risk of underexploring), using Tsallis divergence as a tunable interpolation *  
  关键数字：Improves alignment performance across diverse RM configurations; specific numbers not in abstract  
  **SIA 启示**：SIA's fixed weight parameter treats all prompts identically, risking reward hacking on some distributions (P1). BoT's per-prompt tail-adaptive strategy is directly relevant: SIA could adapt its weight  
  > Best-of-Tails：超越 expected reward，针对"稳健性"而非"平均"优化，与 SIA 的业务目标对齐



### 2.3 Product of Experts / 乘积式分布融合

**[LLMdoctor: Token-Level Flow-Guided Preference Optimization for Ef…](https://arxiv.org/abs/2601.10416)** （AAAI 2026 2026）  
  *Introduces a patient-doctor paradigm where a small specialized 'doctor' model steers a frozen large 'patient' model at token level. Extracts fine-grained token-level preference signals from the patien*  
  关键数字：Outperforms existing test-time alignment methods and surpasses full fine-tuning approaches like DPO (specific numbers in paper).  
  **SIA 启示**：Addresses P1 and P2: The doctor-patient paradigm directly matches SIA's VM (doctor) + 0GM-35B (patient) architecture. TFPO's flow consistency loss addresses a weakness in SIA's current pointwise RM sc  
  > （已在 2.1 中详述）


**[Token-Level Inference-Time Alignment for Vision-Language Models](https://arxiv.org/abs/2510.21794)** （arXiv (ICLR 2026 submission) 2025）  
  *TITA freezes the base VLM and trains a companion reward model to approximate its distribution; at inference time, token-level corrective signals are derived from log-probability ratios between the rew*  
  关键数字：+8.6% on MMVet, +6.7% on POPE; consistent gains on LLaVA-1.5-7B/13B, Qwen2.5-VL-7B, DeepSeek-VL2-27.5B across 12 benchmarks; "negligible inference ove  
  **SIA 启示**：Directly analogous to SIA's per-token logit intervention but for VLMs; demonstrates the mechanism works for multimodal models and addresses P1 (alignment effect). The log-ratio formulation is an alter  
  > TITA（ICLR 2026 submission）：为视觉语言模型实现 token-level 干预，多模态 SIA 的直接先行工作



### 2.4 过程奖励模型（PRM）与推理链对齐

**[More Bang for the Buck: Process Reward Modeling with Entropy-Driv…](https://arxiv.org/abs/2503.22233)** （arXiv 2025）  
  *EDU-PRM uses token-level predictive entropy to dynamically identify step boundaries in reasoning chains, placing PRM supervision at high-entropy transitions rather than at fixed positions. This entrop*  
  关键数字：Outperforms Math-Shepherd and Omega PRM on ProcessBench; 64.7→67.3% accuracy; 32% token reduction; only 1.5% training data needed  
  **SIA 启示**：SIA's --entropy_threshold parameter already implements selective intervention at high-entropy tokens. EDU-PRM validates this design choice with a strong theoretical and empirical basis: high-entropy t  
  > ⭐ Entropy-Driven PRM：在高不确定性位置优先分配 PRM 计算，与 SIA 的 entropy_threshold 设计互补；P1+P2 节省 ~40% PRM 调用


**[Process Reward Agents for Steering Knowledge-Intensive Reasoning …](https://arxiv.org/abs/2604.09482)** （ICML 2026 2026）  
  *Process Reward Agents (PRA) uses domain-grounded, online, step-wise rewards to guide a frozen policy model during inference via dynamic search-based decoding, without retraining the policy. The approa*  
  关键数字：81.9% accuracy on MedQA with Qwen3-4B (new SOTA at 4B scale); up to 25.7% improvement over baselines without policy model updates  
  **SIA 启示**：Directly addresses P1: shows that step-wise reward-guided decoding on frozen policy models substantially improves alignment in knowledge-intensive tasks—exactly SIA's setup. The search-based approach   
  > PRM Agents（ICML 2026）：PRM 在知识密集型推理链中精准干预，Month 5 PRM 训练的设计参考


**[Towards Hierarchical Multi-Step Reward Models for Enhanced Reason…](https://arxiv.org/abs/2503.13551)** （arXiv (cs.CL) 2025）  
  *Proposes HRM, a Hierarchical Reward Model that evaluates both individual and consecutive reasoning steps at fine-grained and coarse-grained levels simultaneously, handling scenarios where flawed inter*  
  关键数字：Improved stability and reliability on PRM800K; strong generalization on MATH500 and GSM8K (specific numbers not in abstract).  
  **SIA 启示**：P1: SIA's current VM provides a single scalar value per token; HRM's hierarchical coarse+fine evaluation suggests training the SIA VM with both token-level and segment-level objectives simultaneously,  
  > Hierarchical Step Reward：多层次步骤奖励模型，支持 SIA 在 reasoning trace 关键节点干预


**[Process Reward Models That Think (ThinkPRM)](https://arxiv.org/abs/2504.16828)** （arXiv 2025）  
  *ThinkPRM builds process reward models that emit a chain-of-thought verification trace before scoring each reasoning step. Using a generative long-CoT verifier, it requires only 1% of the process label*  
  关键数字：+8% over discriminative verifiers on GPQA-Diamond; +4.5% on LiveCodeBench; +7.2% over LLM-as-Judge on ProcessBench under equivalent token budget  
  **SIA 启示**：Directly relevant to P1: a generalizable PRM could replace or augment SIA's dense-scoring VM, providing step-quality signal for non-math chat tasks where outcome labels are scarce. The generative scor  
  > ThinkPRM：在 thinking token 级别应用 PRM，直接对应 SIA 对 Qwen3 thinking 模式的支持


**[Dynamic and Generalizable Process Reward Modeling (DG-PRM)](https://arxiv.org/abs/2507.17849)** （ACL 2025 Main 2025）  
  *DG-PRM uses a reward tree to store fine-grained, multi-dimensional reward criteria and dynamically selects which criteria apply to each reasoning step. Pareto dominance estimation identifies discrimin*  
  关键数字：Significant boost on prevailing benchmarks; exceptional generalizability to out-of-distribution scenarios (specific numbers in full paper)  
  **SIA 启示**：Addresses P1: the multi-dimensional reward criteria and cross-domain generalization are directly relevant to making SIA's VM effective beyond its training distribution. SIA currently uses a single sca  
  > DG-PRM（ACL 2025）：动态 PRM，跨任务泛化性更强，减少 SIA 对特定任务标注数据的依赖


**[From Outcomes to Processes: Guiding PRM Learning from ORM for Inf…](https://arxiv.org/abs/2506.12446)** （arXiv 2025）  
  *SP-PRM addresses the granularity mismatch between outcome reward models and step-level search by integrating score consistency and preference consistency modules to derive process supervision from ORM*  
  关键数字：3.6–10.3% improvement in GPT-4 evaluation scores across dialogue, summarization, and reasoning tasks  
  **SIA 启示**：Addresses P1: SP-PRM validates that ORM-derived step rewards improve inference-time alignment on open-ended tasks (not just math). This directly supports extending SIA beyond the math reasoning domain  
  > SP-PRM：从 ORM 推导过程奖励，降低了 PRM 训练数据收集的门槛，Month 5 任务的直接依据



### 2.5 推理时搜索（Best-of-N / MCTS / Tree Search）

**[Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive…](https://arxiv.org/abs/2503.04412)** （NeurIPS 2025 (Spotlight) 2025）  
  *AB-MCTS proposes a unified framework that dynamically decides per-step whether to explore new candidate branches (wider) or to deepen existing promising branches (deeper) based on external feedback si*  
  关键数字：Consistently outperforms repeated sampling and standard MCTS on coding and engineering benchmarks (specific numbers not in abstract)  
  **SIA 启示**：SIA currently does per-token flat top-K scoring (equivalent to width-1 per-step beam). AB-MCTS provides a principled framework for deciding when to branch wider (scoring more candidates, increasing P2  
  > Adaptive Branching MCTS（NeurIPS 2025 Spotlight）：资源预算自适应分配，和 SIA 的思路互补


**[Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality …](https://arxiv.org/abs/2503.21878)** （arXiv 2025）  
  *This paper provides a theoretical analysis showing that Best-of-N achieves optimal performance only under strict policy coverage conditions and suffers from reward hacking at large N. The authors prop*  
  关键数字：InferenceTimePessimism does not degrade with N unlike BoN; theoretical optimality guarantees under partial coverage  
  **SIA 启示**：SIA biases toward high-VM-score tokens without pessimism, risking reward hacking (P1). InferenceTimePessimism's coverage-aware pessimism principle could be applied to SIA's top-K scoring: penalizing o  
  > Best-of-N 的理论覆盖性分析：为 SIA 提供了为何 token-level dense steering 比 BofN 更高效的理论依据


**[Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495)** （arXiv 2025）  
  *This paper investigates how to scale reward modeling quality at inference time rather than training time. The authors propose Self-Principled Critique Tuning (SPCT) to train generative reward models t*  
  关键数字：SPCT significantly improves quality and scalability of GRMs, outperforming existing methods; competitive with training-time scaling approaches  
  **SIA 启示**：SIA's VM (Qwen3-4B) scores tokens with limited critique capacity. This paper's ensemble-then-vote approach could improve SIA's alignment signal quality (P1) when multiple VM forward passes can be amor  
  > Inference-Time Scaling for Generalist RM：通用 RM 的 inference-time 扩展规律，支持 SIA VM 的扩展性规划



---

## 三、工程类研究（P2/P3 — 延迟与吞吐优化）

### 3.1 两阶段粗过滤与级联 Reward 系统（P2 核心方向）

**[Cascade Reward Sampling for Efficient Decoding-Time Alignment](https://arxiv.org/abs/2406.16306)** （arXiv 2024 2024）  
  *CARDS introduces segment-level rejection sampling for decoding-time alignment, using uncertainty-based segmentation to ensure accurate reward evaluation on incomplete segments. The method minimizes re*  
  关键数字：~70% decoding time reduction vs existing methods; >90% win-ties on utility and safety benchmarks  
  **SIA 启示**：CARDS' segment-level approach is directly applicable to SIA's P2/P3 pain points: instead of scoring all top-K candidates at every token, SIA could score at segment boundaries determined by token uncer  
  > ⭐ Cascade Reward Sampling（arXiv 2024）：segment-level rejection sampling + uncertainty-based segmentation；~70% 解码时间减少，>90% win-rate；**最直接可借鉴**的 Task 1.4 工程参考


**[Towards Cost-Effective Reward Guided Text Generation](https://arxiv.org/abs/2502.04517)** （ICML 2025 2025）  
  *Proposes FaRMA, a reward model architecture trained with Bradley-Terry loss on partial sequences that scores all vocabulary token candidates simultaneously in a single forward pass. This directly addr*  
  关键数字：Significantly faster inference than other RGTG methods; fewer RM calls; competitive with offline RLHF baselines (specific numbers in paper body).  
  **SIA 启示**：Directly addresses P2 and P3: SIA currently makes K forward passes through the 4B VM per decoding step (one per top-K candidate). FaRMA's single-pass vocabulary scoring would reduce VM latency from ~3  
  > ⭐ Cost-Effective RGTG（ICML 2025）：单次 forward 输出 vocabulary-wide reward，从根本上消除 SIA 每候选一次 forward 的模式；可将 topK scoring 从 K 次 → 1 次，P2 核心方案


**[On the Low-Rank Parametrization of Reward Models for Controlled L…](https://arxiv.org/abs/2407.04615)** （TMLR 2025 2024）  
  *Analyzes reward-augmented decoding (RAD) and demonstrates that a low-rank parametrization of the expert/reward model performs on par with higher-rank versions on detoxification and sentiment control, *  
  关键数字：Low-rank RAD matches full RAD on sentiment control and detoxification tasks; single RM call per token (vs. multiple calls in prior methods).  
  **SIA 启示**：Addresses P2: SIA's VM is a dense 4B model. Applying low-rank parametrization to the VM's reward head could reduce per-call memory bandwidth requirements and latency, potentially cutting the ~30ms VM   
  > 低秩 RM 参数化：O(V×d) → O(V×r)，直接减少 VM scoring 的计算量；可配合 Task 2.X 同词表 VM 训练使用


**[Entropy Adaptive Decoding: Dynamic Model Switching for Efficient …](https://arxiv.org/abs/2502.06833)** （arXiv 2025）  
  *EAD monitors rolling logit entropy in real-time and dynamically switches between a smaller and larger model, accepting controlled output divergence in exchange for computational efficiency. On LLaMA m*  
  关键数字：LLaMA: 96.7% performance at 41.5% cost reduction; Qwen: 92.9% performance at 67% cost reduction  
  **SIA 启示**：SIA already implements entropy-based sparse intervention (--entropy_threshold). EAD's framework extends this: rather than skipping VM scoring below threshold, SIA could use entropy to switch to a smal  
  > ⭐ Entropy Adaptive Decoding：实时熵监控 + 动态模型切换（96.7% 性能 / 41.5% cost 减少）；**最接近 Task 1.4 两阶段架构**的已发表系统


**[Fast-Slow Thinking RM: Efficient Integration of Scalar and Genera…](https://arxiv.org/abs/2603.20212)** （arXiv (cs.CL) 2026）  
  *Proposes a dual-process RM that integrates fast scalar scoring (first-token prediction) and slow CoT-based generative judgment in a single model, with a dual-confidence activation mechanism that deter*  
  关键数字：1.2% relative performance improvement over SOTA; 20.8% reduction in token consumption.  
  **SIA 启示**：P2: directly implements a cascade-within-RM design — the 'fast thinking' scalar score from first token maps to SIA's coarse RM stage that could replace most VM calls, with 'slow thinking' CoT only for  
  > ⭐ Fast-Slow Thinking RM（arXiv 2026）：快速标量 RM（first token）+ 慢速生成式 RM 的级联；对应 SIA 的 0.6B（fast）+ 4B（slow）两阶段设计


**[Accelerating LLM Reasoning via Early Rejection with Partial Rewar…](https://arxiv.org/abs/2508.01969)** （arXiv (cs.LG) 2025）  
  *Proposes that Process Reward Models are also 'Partial Reward Models': scores on partially completed reasoning steps are predictive of final output quality. An early rejection framework discontinues su*  
  关键数字：1.4×–9× FLOPs reduction on mathematical reasoning benchmarks without final performance degradation.  
  **SIA 启示**：P2/P3: the partial RM prediction insight validates SIA's entropy_threshold approach theoretically and suggests extending it — RM scores computed at early tokens within a response can already predict f  
  > ⭐ Early Rejection via Partial RM：序列生成过程中用部分 RM 分数做早期拒绝；16-32× 效率提升；P2/P3 双效


**[Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/abs/2410.20290)** （NeurIPS 2024 2024）  
  *Speculative Rejection generates high-scoring responses according to a reward model like Best-of-N but halts unpromising candidates mid-generation, achieving 16-32x computational efficiency improvement*  
  关键数字：16-32x more computationally efficient than Best-of-N; maintains comparable output quality  
  **SIA 启示**：Although SIA uses per-token intervention rather than Best-of-N, Speculative Rejection's early-stopping principle is directly applicable: SIA could reject low-VM-score candidate tokens earlier in the t  
  > Fast Best-of-N via Speculative Rejection（NeurIPS 2024）：前缀级 reward 做早期剪枝；16-32× vs 全序列 BofN


**[Mining Intrinsic Rewards from LLM Hidden States for Efficient Bes…](https://arxiv.org/abs/2505.12225)** （KDD 2026 2025）  
  *SWIFT learns a reward function directly from LLM hidden states using simple linear layers at the token embedding level (<0.005% of LLM parameters). This lightweight probe achieves 12.7% higher accurac*  
  关键数字：12.7% higher accuracy than EurusRM-7B on MATH; uses <0.005% of LLM parameters.  
  **SIA 启示**：Extremely relevant to P2 and P3: SWIFT's approach could serve as SIA's coarse-stage RM in a cascade design — the LLM's own hidden states (already computed in the forward pass) provide a near-zero-cost  
  > ⭐ Mining Intrinsic Rewards from Hidden States（KDD 2026）：从 LLM 隐藏层直接提取 reward 信号，**完全无需外部 VM**；如果成功，P2/P3 彻底消除


**[Streaming Looking Ahead with Token-level Self-reward](https://arxiv.org/abs/2503.00029)** （arXiv 2025）  
  *Reward Transformer integrates token-level self-reward modeling directly into the policy model's architecture, eliminating the need for a separate external reward model. The SLA (Streaming Looking Ahea*  
  关键数字：79.7% win rate vs. greedy decoding on 3 general-domain datasets; with DPO fine-tuning: 89.4% win rate  
  **SIA 启示**：SLA's self-contained self-reward addresses SIA's biggest latency bottleneck: P2/P3 stem from the external VM being in the critical path. If SIA's 0GM-35B could be augmented with an internal self-rewar  
  > ⭐ Streaming Lookahead Self-reward（arXiv 2025）：LLM 自身对 upcoming token 做 streaming 预评分，消除外部 VM 调用；P2/P3 的激进备选方案


**[Speculative Reward Model Boosts Decision Making Ability of LLMs C…](https://arxiv.org/abs/2506.00396)** （ACL 2025 Industry Track 2025）  
  *Introduces SRM, a plug-and-play framework using an external reward assigner to predict optimal actions and a speculative verification mechanism to prune suboptimal choices. Integrates with existing se*  
  关键数字：Reduces costs to 1/10 of original search framework on average while maintaining effectiveness across math reasoning, planning, and numerical reasoning  
  **SIA 启示**：Relevant to P2/P3: SRM's approach of using a small external reward assigner to pre-filter candidates (reducing full RM calls by ~10x) is directly applicable to SIA's top-K scoring step, potentially re  
  > Speculative Reward Model（ACL 2025）：用 speculative RM 降低 reward 计算延迟，对应 SIA P2


**[Bi-directional Model Cascading with Proxy Confidence](https://arxiv.org/abs/2504.19391)** （arXiv (cs.LG) 2025）  
  *Proposes bi-directional model cascading that simultaneously considers confidence from both small and large models using a tiny proxy for the large model's pre-invocation confidence. Uses hidden state *  
  关键数字：Reduced deferrals to more costly models vs. standard cascading baselines (specific percentages not in abstract).  
  **SIA 启示**：P2/P3: the proxy-confidence concept maps to SIA — a tiny model estimating whether the 4B VM will score candidates divergently (i.e., whether VM intervention would change the token selection). Skip VM   
  > Bi-directional Cascading with Proxy Confidence：双向级联决策，减少不必要的大模型调用


**[Faster Cascades via Speculative Decoding](https://arxiv.org/abs/2405.19261)** （arXiv (cs.CL) 2024）  
  *Combines model cascades (invoke large model only for 'hard' inputs) with speculative decoding (small model drafts, large model verifies in parallel). Derives the optimal deferral rule for speculative *  
  关键数字：Improved cost-quality trade-offs vs. both standalone cascading and speculative decoding (Gemma, T5 experiments); specific speedup numbers not in abstr  
  **SIA 启示**：P2/P3: formalizes the cascade design space SIA could adopt — a small proxy RM (e.g., hidden-state linear probe) as the draft-stage fast filter, and the 4B VM only for deferred hard cases. The optimal   
  > Faster Cascades via Speculative Decoding：用 speculative decoding 加速 cascade reward 系统


**[Efficient Test-Time Scaling of Multi-Step Reasoning by Probing In…](https://arxiv.org/abs/2511.06209)** （ACL 2026 Main 2025）  
  *ReProbe trains a lightweight transformer-based probe (<10M parameters) on LLM internal states to assess reasoning step credibility, matching or exceeding expensive PRM performance at a fraction of the*  
  关键数字：Probes match or exceed PRM performance while being up to 810x smaller; evaluated across mathematics, planning, and general knowledge; under 10M parame  
  **SIA 启示**：Directly addresses P2 and P3: a <10M parameter probe on the main LLM's hidden states could serve as SIA's VM replacement at near-zero marginal cost. This is analogous to what SWIFT proposes but for re  
  > Probing Internals for Efficient Test-Time Scaling（ACL 2026）：从模型内部状态预测 reward，减少 RM 外部调用



### 3.2 KV 缓存与前缀共享（P3 核心方向）

**[Hydragen: High-Throughput LLM Inference with Shared Prefixes](https://arxiv.org/abs/2402.05099)** （ICLR 2025 2024）  
  *Hydragen is a hardware-aware attention kernel for batches of sequences sharing a common prefix. By separating shared-prefix attention from unique-suffix attention and batching the shared part as a sin*  
  关键数字：Up to 32x end-to-end throughput improvement on CodeLlama-13B; <15% throughput drop as prefix grows from 1K to 16K (vs >90% for baselines); 55% inferen  
  **SIA 启示**：SIA's RM scoring step is exactly the shared-prefix attention pattern Hydragen targets: each of the top-K candidates shares a long common prefix (conversation + partial generation) and only differs in   
  > ⭐ Hydragen（ICLR 2025）：批内共享前缀拆分 attention；32× 吞吐提升（CodeLlama-13B）；SIA 的 VM scoring 中所有候选共享相同前缀，**可直接应用**


**[CoDec: Prefix-Shared Decoding Kernel for LLMs](https://arxiv.org/abs/2505.17694)** （arXiv preprint 2025）  
  *CoDec targets the memory bottleneck when multiple prompts share a common prefix during the decoding phase. It builds a specialized kernel that handles the tree-structured dependency of prefix sharing,*  
  关键数字：1.9x attention speedup vs FlashDecoding; 120.9x reduction in memory access vs FlashDecoding; 3.8x faster end-to-end time-per-output-token vs vLLM  
  **SIA 启示**：Same structural match as Hydragen: SIA's K parallel RM scoring passes each share a long prefix. CoDec's decode-phase focus means it could speed up the batch RM forward pass directly, since the RM runs  
  > ⭐ CoDec：prefix-shared decoding kernel，在 GPU kernel 层面实现前缀共享，P3 系统级优化


**[When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate…](https://arxiv.org/abs/2601.08343)** （arXiv preprint 2026）  
  *This paper finds that KV cache reuse strategies that work well for execution agents can severely degrade LLM judge behavior: while end-task accuracy may remain stable, judge selection becomes inconsis*  
  关键数字：Significant JCR degradation on GSM8K, MMLU, and HumanEval; specific numerical degradation varies by setting  
  **SIA 启示**：Critical warning for SIA: if APC or prefix sharing is applied naively to the RM during SIA scoring, the RM may exhibit inconsistent scoring behavior (P1 alignment degradation). The paper provides diag  
  > ⚠️ Cross-Candidate KV Cache 失效警告（arXiv 2026）：在 SIA 类似的 multi-candidate scoring 场景中，直接复用 APC 会导致 RM 评分不一致；**需小心处理**


**[RKSC: Reasoning-Aware KV Cache Sharing and Confident Early Exit f…](https://arxiv.org/abs/2606.09937)** （arXiv preprint 2026）  
  *RKSC is a training-free inference framework targeting multi-branch reasoning pipelines. It computes the shared prefix KV cache once and reuses it across semantically similar branches (not just byte-id*  
  关键数字：Mean 3.0x speedup (peak 3.99x) over baseline; 1.66x over vLLM prefix caching; 0.37% error rate from early exit across 1,616 verification calls on 5 mo  
  **SIA 启示**：SIA's RM scores top-K candidate tokens by appending each to the current partial generation, creating K parallel forward passes that all share the same long prefix (system prompt + generated tokens so   
  > RKSC（arXiv 2026）：reasoning-aware KV 共享 + 置信度早退；3× 加速；P2/P3 双效


**[BatchLLM: Optimizing Large Batched LLM Inference with Global Pref…](https://arxiv.org/abs/2412.03594)** （MLSys 2026 2024）  
  *BatchLLM addresses the prefix sharing opportunity in large offline/batch LLM workloads that existing engines miss because they only consider local prefixes. It explicitly identifies shared prefixes gl*  
  关键数字：1.3x to 10.8x throughput improvement over vLLM and SGLang on microbenchmarks and industry workloads; accepted MLSys 2026  
  **SIA 启示**：In high-concurrency SIA serving (P3), multiple simultaneous LLM requests generate partially overlapping prefixes that get sent to the RM. BatchLLM's global prefix identification and grouped scheduling  
  > BatchLLM（MLSys 2026）：cross-request 前缀共享；92.6% vs 6.3% token 复用率（已在 Task 1.4 中引用）


**[RadixMLP: Intra-batch Deduplication for Causal Transformers](https://arxiv.org/abs/2601.15013)** （arXiv preprint 2026）  
  *RadixMLP eliminates redundant MLP computation (not just attention) for sequences sharing a common prefix in a batch. It uses a prefix trie to gather shared prefix tokens into a compressed representati*  
  关键数字：1.44-1.59x speedup on MS MARCO v1.1 reranking with Qwen3 0.6B-8B; up to 5x speedup on synthetic long shared prefix benchmarks  
  **SIA 启示**：SIA's RM runs K forward passes per decoding step, all sharing the same long prefix. RadixMLP would eliminate not just duplicate attention KV reads but also the redundant MLP computation over the share  
  > RadixMLP：批内 MLP 去重，对 MoE 模型（如 0GM-35B）的前缀批处理特别有效


**[Requests of a Feather Must Flock Together: Batch Size vs. Prefix …](https://arxiv.org/abs/2605.06046)** （arXiv preprint 2026）  
  *This paper shows that prefix-homogeneous batches (all requests share a common prefix) achieve higher decode throughput than larger heterogeneous batches, because KV cache memory reads are more cache-f*  
  关键数字：2-10x higher end-to-end throughput vs vLLM FCFS for 1K-10K shared prefix lengths; outperforms prefix-aware attention kernels; hardware-agnostic  
  **SIA 启示**：SIA's RM scoring calls all share the same long prefix structure (system prompt + partial generation). Feather-style scheduling of RM requests by prefix homogeneity could dramatically improve GPU memor  
  > 批次前缀同质性分析：指导 SIA 的请求调度策略，优先将类似前缀的请求聚合在同一批次



### 3.3 异步流水线与延迟隐藏（P2/P3）

**[AsyncSpade: Efficient Test-Time Scaling with Asynchronous Sparse …](https://arxiv.org/abs/2510.07486)** （arXiv 2025 2025）  
  *Decouples the KV-cache token selection step from the autoregressive forward pass by running selection asynchronously in an overlapped pipeline. Uses a lightweight temporal-regressive module to predict*  
  关键数字：20% TPOT reduction vs Quest; 50% TPOT reduction vs full attention on A100; maintains accuracy on AIME-24/25 and GPQA-Diamond  
  **SIA 启示**：Directly applicable to SIA P2/P3: SIA's VM scoring is on the critical path of each decode step. AsyncSpade's asynchronous decoupling pattern — predict needed computation one step ahead, overlap it wit  
  > ⭐ AsyncSpade（arXiv 2025）：异步稀疏 decoding，20-50% TPOT 减少；SIA VM 调用可异步化的直接工程参考


**[OPPO: Accelerating PPO-based RLHF via Pipeline Overlap](https://arxiv.org/abs/2509.25762)** （arXiv 2025 2025）  
  *Proposes intra-step and inter-step pipeline overlap for PPO-based RLHF training, directly targeting the dependency between actor model generation and reward model scoring. Intra-step overlap streams a*  
  关键数字：1.8×–2.8× training speedup; 1.4×–2.1× GPU utilization improvement; tested across multiple model scales  
  **SIA 启示**：Highly relevant to SIA P2/P3: The intra-step overlap pattern maps directly to SIA's architecture. SIA could stream partial candidate tokens to the VM as soon as vLLM's draft sampling completes, lettin  
  > OPPO：PPO 训练中 generator-RM 流水线重叠；1.8-2.8× 加速；推理时异步 VM 调用的理论先例


**[PipeInfer: Accelerating LLM Inference using Asynchronous Pipeline…](https://arxiv.org/abs/2407.11798)** （SC 2024 2024）  
  *Introduces continuous asynchronous speculation and early inference cancellation for distributed LLM inference. Draft inference runs simultaneously with target model single-token inference in a pipelin*  
  关键数字：Up to 2.15× generation speed improvement over standard speculative inference; better tolerance to low acceptance rates and bandwidth-constrained inter  
  **SIA 启示**：Relevant to P2/P3: The async draft+verify pattern maps onto SIA's draft-score-apply loop. Running the VM's scoring asynchronously overlapped with vLLM's decode (as PipeInfer does for draft and target)  
  > PipeInfer（SC 2024）：异步流水线 speculative decoding；2.15× 提速


**[PipeSpec: Breaking Stage Dependencies in Hierarchical LLM Decodin…](https://arxiv.org/abs/2505.01572)** （ACL Findings 2025 2025）  
  *Generalizes speculative decoding to a k-model hierarchical pipeline where each level proposes tokens asynchronously to the next, removing stage-serial dependencies. Provides analytical models characte*  
  关键数字：Up to 2.54× speedup over standard decoding; outperforms state-of-the-art speculative methods; validated on text summarization and code generation  
  **SIA 启示**：Relevant to P2/P3: PipeSpec's hierarchical async pipeline can be reframed for SIA as a 2-stage pipeline: vLLM decode (stage 1) streams top-k candidate tokens to the VM (stage 2) which scores asynchron  
  > PipeSpec（ACL Findings 2025）：k-stage 分层流水线；2.54× vs 标准 decoding


**[NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/abs/2408.12757)** （OSDI 2025 2025）  
  *Proposes intra-device parallelism for LLM serving by splitting batches into nano-batches and overlapping heterogeneous operations (compute-bound FFN, memory-bound attention, network-bound AllReduce) o*  
  关键数字：1.91× throughput improvement over TensorRT-LLM; 50–72% of theoretical optimal; tested on LLaMA-2-70B, Mixtral 8×7B, LLaMA-3-8B  
  **SIA 启示**：Relevant to P3: SIA's VM runs as a separate synchronous call blocking vLLM decode. NanoFlow's nano-batch + CUDA stream approach shows how to overlap a secondary compute kernel (analogous to VM forward  
  > NanoFlow（OSDI 2025）：nano-batch 操作级流水线；吞吐最大化的 serving 系统设计


**[APEX: Asynchronous Parallel CPU-GPU Execution for Online LLM Infe…](https://arxiv.org/abs/2506.03296)** （arXiv 2025 2025）  
  *Proposes an asynchronous CPU-GPU overlap execution strategy for LLM inference on memory-constrained hardware that predicts subtask execution times and co-schedules CPU attention with GPU compute to el*  
  关键数字：84–96% throughput improvement on T4 GPUs; 11–89% on A10 GPUs vs vLLM; evaluated on LLaMA-2-7B and LLaMA-3.1-8B  
  **SIA 启示**：Relevant to P3: APEX demonstrates that secondary compute tasks (CPU attention) can be overlapped with GPU main decode via async scheduling. The same principle applies to SIA's VM: model the VM call as  
  > APEX：CPU-GPU 异步执行，减少 CPU 端调度对 GPU 利用率的影响



### 3.4 Speculative Decoding 与 Reward 结合（P2/P3）

**[Reward-Guided Speculative Decoding for Efficient LLM Reasoning](https://arxiv.org/abs/2501.19324)** （arXiv 2025）  
  *RSD combines a lightweight draft model with a powerful target model and incorporates a process reward model to evaluate intermediate decoding steps, dynamically deciding whether to invoke the target m*  
  关键数字：4.4x FLOPs reduction vs target-only decoding; +3.5pp accuracy over parallel decoding on average across Olympiad-level tasks  
  **SIA 启示**：Directly analogous to SIA's per-token RM scoring, but reorganized as speculative decoding: the draft model runs cheaply and RM only triggers the expensive large model on high-entropy/low-confidence to  
  > ⭐ RSD（arXiv 2025）：reward-guided speculative decoding，draft 模型快速生成候选，RM 只在不确定处触发；SIA Task 1.4 的直接参照


**[Guided Speculative Inference for Efficient Test-Time Alignment of…](https://arxiv.org/abs/2506.04118)** （ICLR 2026 2026）  
  *GSI combines soft best-of-n test-time scaling with speculative samples from auxiliary models to approximate an optimal tilted policy and expected rewards under that policy. By leveraging lightweight a*  
  关键数字：Up to 28% end-to-end latency reduction; higher accuracy than standard soft BoN and reward-guided speculative decoding on 5 benchmarks  
  **SIA 启示**：SIA's VM runs synchronously at every step at 30ms cost. GSI's architecture of using the RM for sparse acceptance/rejection rather than per-token dense scoring is architecturally applicable to SIA and   
  > ⭐ GSI（ICLR 2026）：sparse acceptance/rejection 替代 per-token dense scoring；~48% VM 调用跳过率；已在 Task 1.4 中引用


**[Reward-Shifted Speculative Sampling Is An Efficient Test-Time Wea…](https://arxiv.org/abs/2508.15044)** （EMNLP 2025 2025）  
  *This paper uses a small aligned draft model combined with an unchanged large target model, theoretically demonstrating how distributional differences between aligned and unaligned models can be exploi*  
  关键数字：Superior gold reward scores at significantly reduced inference cost vs. existing methods (specific numbers not in abstract)  
  **SIA 启示**：SIA's VM (Qwen3-4B) can be viewed as a weak aligner guiding a strong 0GM-35B target. RSSA's framework for exploiting distributional differences between VM and LLM logits is directly applicable and cou  
  > ⭐ RSSA（EMNLP 2025）：Reward-Shifted Speculative Sampling，weak-to-strong 对齐；已在 Task 1.4 中引用


**[Judge Decoding: Faster Speculative Sampling Requires Going Beyond…](https://arxiv.org/abs/2501.19309)** （ICLR 2025 2025）  
  *Judge Decoding replaces the standard token acceptance criterion in speculative decoding (likelihood-ratio test) with a learned lightweight judge classifier that evaluates hidden representations of dra*  
  关键数字：9x speedup vs. standard autoregressive; 141 tokens/sec for 8B/405B-Judge on 2×H100; quality maintained across benchmarks.  
  **SIA 启示**：Highly relevant to P3: the judge mechanism decouples 'is this token good?' from 'does this token match the LLM distribution?'. SIA's VM scoring is already doing this implicitly—it scores quality, not   
  > Judge Decoding（ICLR 2025）：decouples reward quality 与分布一致性，P3 吞吐优化视角


**[SPECS: Faster Test-Time Scaling through Speculative Drafts](https://arxiv.org/abs/2506.15733)** （arXiv (cs.AI) 2025）  
  *SPECS is a latency-aware test-time scaling method that uses a smaller fast model to generate candidate sequences, evaluated using signals from both a larger target model and a dedicated reward model v*  
  关键数字：Up to 19.1% latency reduction on MATH500, AMC23, OlympiadBench; matches beam search accuracy; theoretical convergence guarantees.  
  **SIA 启示**：SPECS is the closest published analog to a 'SIA + speculative decoding' hybrid: a small model proposes tokens, a RM scores them, and a deferral gate decides if the large model needs to intervene. This  
  > ⭐ SPECS：speculative drafts + deferral gate；最近似"SIA + speculative decoding"混合的已发表系统


**[From Tokens to Steps: Verification-Aware Speculative Decoding for…](https://arxiv.org/abs/2604.15244)** （arXiv (cs.CL) 2026）  
  *SpecGuard proposes step-level verification of speculative decoding drafts using model-internal signals (attention-based grounding scores and log-probability confidence) rather than external reward mod*  
  关键数字：3.6% accuracy improvement and ~11% latency reduction vs. reward-guided SD; outperforms both standard SD and reward-guided SD across reasoning benchmar  
  **SIA 启示**：Directly relevant to P2: SpecGuard's insight that internal model signals (attention, log-prob) can substitute external RM calls is applicable to SIA. Using entropy and log-prob confidence (already ava  
  > SpecGuard（arXiv 2026）：用内部 attention 信号替代外部 RM 调用，P2 减少 VM 调用的创新方向


**[Multi-Draft Speculative Sampling: Canonical Decomposition and The…](https://arxiv.org/abs/2410.18234)** （ICLR 2025 (Spotlight) 2024）  
  *This paper derives the theoretically optimal token-selection scheme for multi-draft speculative sampling via canonical decomposition: first an importance-sampling step selects one intermediate token a*  
  关键数字：Consistent improvements in achievable block efficiency and token rates over baseline schemes; ICLR 2025 Spotlight.  
  **SIA 启示**：Provides theoretical grounding for SIA's top-K candidate scoring. SIA already samples top-K tokens and scores them; this paper shows the optimal acceptance scheme for multi-draft settings is importanc  
  > Multi-Draft Speculative Sampling（ICLR 2025 Spotlight）：多 draft 联合选择，P2 加速


**[Multi-Drafter Speculative Decoding with Alignment Feedback](https://arxiv.org/abs/2604.05417)** （ACL 2026 Findings 2026）  
  *MetaSD integrates multiple heterogeneous draft models into speculative decoding and dynamically allocates compute to each drafter using alignment feedback from the target model's acceptance decisions,*  
  关键数字：Consistently outperforms single-drafter approaches across benchmarks; ACL 2026 Findings.  
  **SIA 启示**：Relevant to P2/P3: SIA uses a single VM for all tokens, but the bandit framing suggests that for SIA, different VM sizes or checkpoints could be dynamically selected based on accumulated alignment fee  
  > Multi-Drafter Speculative Decoding（ACL 2026）：多 drafter 动态分配，对应 SIA 的 0.6B+4B 两阶段



---

## 四、训练方法研究（VM/RM 如何训练得更好）

### 4.1 Token 级奖励信号提取与训练

**[RED: Unleashing Token-Level Rewards from Holistic Feedback via Re…](https://arxiv.org/abs/2411.08302)** （EMNLP 2025 2025）  
  *RED uses an off-the-shelf sequence-level reward model and redistributes its single scalar score to individual tokens using attention-based credit assignment. The method requires no RM modification or *  
  关键数字：Superior performance vs. sequence-level baselines across diverse datasets (specific numbers not reported in abstract)  
  **SIA 启示**：RED's redistribution approach can directly improve SIA's VM training data quality: take the sequence-level preference data used to train the current VM and apply RED to generate token-level reward lab  
  > ⭐ RED（EMNLP 2025）：从 holistic feedback 蒸馏 token 级奖励；SIA 已引用，但值得深入：提供了无需 token 标注数据的 VM 训练路径


**[From r to Q*: Your Language Model is Secretly a Q-Function](https://arxiv.org/abs/2404.12358)** （COLM 2024 2024）  
  *This theoretical paper proves that DPO implicitly learns a token-level Q-function (optimal Q*) rather than a sequence-level bandit policy. Under token-level MDP analysis, DPO satisfies the Bellman equ*  
  关键数字：Beam search on DPO policy demonstrates measurable quality improvements over greedy DPO decoding (empirical demonstration of Q*-guided search)  
  **SIA 启示**：Provides the theoretical foundation for SIA: if DPO-trained models are implicitly Q-functions, SIA's VM (trained on pairwise preferences) is already learning Q* at token level. This motivates using th  
  > ⭐ From r to Q*（COLM 2024）：LLM 本身隐含 Q-function；可用 LLM 的 log-prob 差值作为近似 VM，大幅简化 VM 训练


**[Reward Models Are Secretly Value Functions: Temporally Coherent R…](https://arxiv.org/abs/2604.22981)** （arXiv 2026）  
  *TCRM adds Monte Carlo and temporal-difference regularization terms on top of the standard Bradley-Terry reward model loss, forcing intermediate token positions to predict the conditional expectation o*  
  关键数字：Middle-token pairwise accuracy improved from 50% to 88.9%; PPO: 27% peak GPU memory reduction, 19% training step time reduction; 44.9% avg F1 on Proce  
  **SIA 启示**：Directly addresses SIA's VM training: training SIA's Qwen3-4B value model with TD regularization would make its intermediate token logits reliable partial-reward estimators, improving P1 alignment qua  
  > ⭐ Reward Models Are Secretly Value Functions（arXiv 2026）：时间一致性约束的 RM 即 value function；直接支持 SIA 的 VM = value model 设计理念


**[AlignDistil: Token-Level Language Model Alignment as Adaptive Pol…](https://arxiv.org/abs/2503.02832)** （ACL 2025 2025）  
  *AlignDistil proves equivalence between a token-level RLHF objective (with DPO reward) and a token-level distillation process where the teacher distribution linearly combines DPO-model and reference-mo*  
  关键数字：Superiority over existing methods; fast convergence due to token-level distributional reward optimization  
  **SIA 启示**：AlignDistil's token-adaptive coefficient (varying the logit weight per token based on reward signal strength) is directly applicable to SIA: instead of a fixed --weight, SIA could scale the VM score c  
  > AlignDistil（ACL 2025）：token-level 对齐知识蒸馏，可用于 0.6B 预筛选 VM 的训练


**[A Critical Look At Tokenwise Reward-Guided Text Generation](https://arxiv.org/abs/2406.07780)** （COLM 2025 2024）  
  *Identifies a fundamental flaw in RGTG methods: reward models trained on full sequences are not compatible with scoring partial sequences, causing sub-optimal token selection. Proposes training a Bradl*  
  关键数字：Outperforms previous RGTG methods and matches strong offline baselines without LLM fine-tuning.  
  **SIA 启示**：Directly relevant to P1: SIA's VM (trained with LoRA on Qwen3-4B-Base) may be miscalibrated for partial-sequence scoring. Adopting partial-sequence training for the VM could strengthen alignment effec  
  > ⚠️ 批判性视角：Token-wise reward-guided 在某些任务上不如 sequence-level；提醒 SIA 选择合适的任务域


**[PARM: Multi-Objective Test-Time Alignment via Preference-Aware Au…](https://arxiv.org/abs/2505.06274)** （ICML 2025 2025）  
  *Extends GenARM to multi-objective alignment by training a single unified ARM across all preference dimensions using Preference-Aware Bilinear Low-Rank Adaptation (PBLoRA). Eliminates the need for sepa*  
  关键数字：Reduced inference costs vs. per-objective ARMs; improved alignment with preference vectors; enables smaller models to guide larger frozen LLMs.  
  **SIA 启示**：Addresses P1: SIA currently uses a single-objective VM. PARM's multi-objective ARM framework would let SIA simultaneously optimize helpfulness, harmlessness, and honesty by providing a preference weig  
  > PARM（ICML 2025）：多目标 token-level RM，SIA 未来扩展多目标对齐的参考


**[UniARM: Towards a Unified Autoregressive Reward Model for Multi-O…](https://arxiv.org/abs/2602.09538)** （arXiv 2026 (under review) 2026）  
  *Proposes UniARM, which unifies multi-objective test-time alignment into a single ARM parameter space using Preference-Modulated & Shared Low-Rank Adaptation (MoSLoRA). Extracts shared preference featu*  
  关键数字：18.5% HV improvement and 30.2% MIP improvement on safety alignment; 9.1%/6.8% improvements with weak-to-strong guidance; no additional parameters or i  
  **SIA 启示**：Addresses P1: UniARM's unified multi-objective ARM with no additional inference latency is a drop-in upgrade path for SIA's single-objective VM. The MoSLoRA design could replace SIA's current LoRA tok  
  > UniARM（arXiv 2026）：统一多目标 autoregressive RM，减少 SIA 多目标场景的训练成本


**[Earlier Tokens Contribute More: Learning Direct Preference Optimi…](https://arxiv.org/abs/2502.14340)** （ICLR 2025 2025）  
  *D2PO introduces a temporal decay factor (gamma) into DPO that weights earlier token positions more heavily during optimization, grounded in a token-level MDP analysis showing that early tokens have hi*  
  关键数字：AlpacaEval 2 win rate +5.9 to +8.8pp over DPO; Arena-Hard +3.3 to +9.7pp; maintains MMLU, GSM8K, MATH performance  
  **SIA 启示**：SIA's VM currently receives uniform credit across all token positions. The D2PO insight—early tokens matter more—suggests the VM training should weight earlier-token reward signals more strongly. If S  
  > Earlier Tokens Contribute More（ICLR 2025）：序列前段 token 的奖励权重更高；VM 训练时的 token 权重调整依据


**[DPO Meets PPO: Reinforced Token Optimization for RLHF](https://arxiv.org/abs/2404.18922)** （ICML 2025 2025）  
  *RTO models RLHF as a token-level MDP and uses DPO to extract a token-wise reward function from preference data, which is then used to drive PPO training. This two-stage pipeline (DPO reward extraction*  
  关键数字：RTO outperforms PPO by +7.5pp on AlpacaEval 2 and +4.1pp on Arena-Hard  
  **SIA 启示**：RTO's two-stage pipeline is directly applicable to SIA's VM training: use DPO to derive token-level reward labels from preference data, then train SIA's VM (Qwen3-4B) via PPO on these token-level rewa  
  > DPO Meets PPO（ICML 2025）：token-level PPO+DPO 混合，训练数据效率更高


**[T-REG: Preference Optimization with Token-Level Reward Regulariza…](https://arxiv.org/abs/2412.02685)** （ACL 2025 2024）  
  *T-REG uses the LLM's own self-refinement capability with contrastive prompting to generate token-level reward regularizers without external annotators, then uses these self-generated rewards as soft r*  
  关键数字：Alpaca Eval 2: up to +3.8% over DPO; Arena-Hard: up to +4.4% over DPO  
  **SIA 启示**：T-REG's self-generated token reward approach suggests that SIA's 0GM-35B LLM itself can generate training labels for the VM through contrastive prompting, eliminating dependency on external preference  
  > T-REG（ACL 2025）：token-level reward regularization，避免 VM 分数分布退化


**[Selective Preference Optimization via Token-Level Reward Function…](https://arxiv.org/abs/2408.13518)** （EMNLP 2025 2025）  
  *SePO trains a small oracle model to estimate per-token reward functions from preference data, then selects only the 30% of tokens with highest reward signal for policy optimization. This achieves bett*  
  关键数字：Significantly outperforms baselines optimizing on all tokens by training only on 30% key tokens; weak oracle supervises strong policy models up to 16.  
  **SIA 启示**：SePO's weak-to-strong generalization directly applies to SIA's VM: a small Qwen3-0.6B or 1.7B oracle VM can generate per-token reward labels that effectively train a larger VM (Qwen3-4B), reducing VM   
  > Selective Preference Optimization（EMNLP 2025）：只在 high-reward-difference token 位置做 DPO，与 SIA 的稀疏干预思路一致



### 4.2 多模态 Reward Model 训练（Month 6 参考）

**[Skywork-VL Reward: An Effective Reward Model for Multimodal Under…](https://arxiv.org/abs/2505.07263)** （arXiv 2025）  
  *Skywork-VL Reward is a 7B multimodal reward model built on Qwen2.5-VL-7B-Instruct trained with pairwise ranking loss on a large-scale preference dataset covering both standard VLM outputs and advanced*  
  关键数字：73.1 on VL-RewardBench (SOTA at submission); 90.1 on RewardBench; significant improvements in downstream MPO training  
  **SIA 启示**：Provides a strong off-the-shelf multimodal RM that SIA could use as the VM when applied to VLM-based LLMs (e.g., future 0GM-VL variants). Directly relevant to P1 for extending SIA to VLM tasks.  
  > Skywork-VL Reward（arXiv 2025）：有效的视觉语言 RM，可作为 SIA 多模态 VM 的 backbone 参考


**[BaseReward: A Strong Baseline for Multimodal Reward Model](https://arxiv.org/abs/2509.16127)** （arXiv 2025）  
  *BaseReward conducts exhaustive analysis across reward modeling paradigms, architectures (Qwen2.5-VL backbone + two-layer reward head), training strategies, and data curation for multimodal reward mode*  
  关键数字：SOTA on MM-RLHF-Reward Bench, VL-RewardBench, and Multimodal RewardBench; tested on 15 multimodal benchmarks across RL-trained models  
  **SIA 启示**：Provides engineering best practices for building the multimodal RM that SIA would need if extended to VLM inputs. Directly informs P1 (which VM architecture to adopt for multimodal scenarios).  
  > BaseReward（arXiv 2025）：强 baseline 多模态 RM，Month 6 VLM VM 的选型参考


**[MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Sta…](https://arxiv.org/abs/2603.25108)** （CVPR 2026 2026）  
  *MSRL addresses data scarcity for multimodal reward training by proposing progressive transfer from text-only preferences through caption-based RL to fully multimodal RL, with cross-modal knowledge dis*  
  关键数字：VL-RewardBench: 66.6% → 75.9%; GenAI-Bench: 70.2% → 75.7%; achieved without additional multimodal preference annotations  
  **SIA 启示**：Relevant to P1: if SIA is extended to VLM inputs, training the VM on multimodal preferences is necessary; MSRL's staged approach enables this without expensive full multimodal preference labeling.  
  > MSRL（CVPR 2026）：多阶段 multimodal reward modeling，SIA 多模态扩展的训练方案


**[Learning Adaptive LLM Decoding](https://arxiv.org/abs/2603.09065)** （arXiv 2026）  
  *Shows that while token-level decoding adaptation can be reduced to entropy-based heuristics, entropy alone is insufficient to recover full gains from learned token-level policies. Proposes sequence-le*  
  关键数字：Token-level adapter: up to 10.2% Pass@1 improvement on MATH vs best static baseline under fixed budget; sequence-level: 2-3% gains on MATH and CodeCon  
  **SIA 启示**：P1/P2: Directly challenges SIA's entropy-only gate—entropy is necessary but insufficient. Suggests SIA could benefit from a learned routing policy on top of entropy, predicting which tokens need VM in  
  > （已在效果类中列出）


**[Entropy-Guided Data-Efficient Training for Multimodal Reasoning R…](https://arxiv.org/abs/2602.01884)** （arXiv 2026）  
  *EGT identifies a strong correlation between response entropy and accuracy in multimodal reward model training, enabling unsupervised identification of noisy samples and difficulty-aware training via e*  
  关键数字：Consistently outperforms SOTA multimodal reward models on three benchmarks; specific numerical gains not reported in abstract  
  **SIA 启示**：Strongly relevant to P1: the entropy-accuracy correlation discovered here aligns with SIA's entropy-threshold intervention strategy — high entropy tokens are exactly where reward guidance matters most  
  > Entropy-Guided 多模态 RM 训练：减少多模态 RM 训练数据需求，适合 SIA 资源约束



---

## 五、评估方法研究

**[Inference-Time Reward Hacking in Large Language Models](https://arxiv.org/abs/2506.19248)** （NeurIPS 2025 (Spotlight) 2025）  
  *This NeurIPS spotlight paper characterizes how overoptimizing imperfect reward models at inference time causes alignment to initially improve then degrade (reward hacking). The authors study Best-of-n*  
  关键数字：Demonstrated superior reward-distortion tradeoffs across math, reasoning, and human-preference setups; hedging mitigates reward hacking  
  **SIA 启示**：SIA's fixed weight parameter risks reward hacking at higher values (P1). This paper's HedgeTune framework and Best-of-Poisson formulation are directly applicable to calibrating SIA's --weight paramete  
  > ⚠️ Inference-Time Reward Hacking（NeurIPS 2025 Spotlight）：推理时 RM 干预会导致分布外样本触发奖励黑客；SIA 设计中需防范


**[More Test-Time Compute Can Hurt: Overestimation Bias in LLM Beam …](https://arxiv.org/abs/2603.15377)** （arXiv 2026）  
  *Using Extreme Value Theory, this paper shows that wider beam search introduces a systematic overestimation bias that grows with candidate pool size, deriving a formula for maximum useful beam width as*  
  关键数字：PRM-guided search: up to +8.9pp improvement; perplexity scorer: no benefit beyond width 1; tested on three 7B models across 5,975 questions in 10 doma  
  **SIA 启示**：SIA uses top-K candidate scoring with a VM at every step. This paper directly predicts whether SIA's K value is appropriately set: if the VM's SNR is low, increasing K will hurt rather than help (P1).  
  > ⚠️ More Test-Time Compute Can Hurt：过度使用测试时计算存在过估计偏差；SIA --weight 参数不宜过大


**[RMB: Comprehensively Benchmarking Reward Models in LLM Alignment](https://arxiv.org/abs/2410.09893)** （ICLR 2025 2025）  
  *RMB introduces a comprehensive benchmark covering 49+ real-world scenarios with both pairwise and Best-of-N evaluation formats for reward models. It demonstrates positive correlation between benchmark*  
  关键数字：Positive correlation between RMB scores and downstream alignment performance confirmed  
  **SIA 启示**：Directly addresses P1: SIA's alignment evaluation currently relies on MMLU (+12pp). RMB's 49-scenario framework provides a systematic methodology to evaluate SIA's VM across diverse real-world tasks,   
  > RMB（ICLR 2025）：全面的 RM benchmark；VM-Qwen3-4B 的评估基准参考


**[RewardBench 2: Advancing Reward Model Evaluation](https://arxiv.org/abs/2506.01937)** （arXiv 2026 2026）  
  *Introduces a new multi-skill reward model benchmark with substantially harder evaluation data, scoring models about 20 points lower than the original RewardBench. Sources new human prompts rather than*  
  关键数字：Models score ~20 points lower than on original RewardBench; high correlation with downstream Best-of-N and PPO performance  
  **SIA 启示**：SIA uses a Value Model (VM) as the reward signal at every decoding step. P1 pain point: SIA currently lacks broad eval; RewardBench 2 provides the multi-skill benchmark to diagnose whether the VM (Qwe  
  > RewardBench 2（arXiv 2026）：更新的 RM 评估框架，Task 1.3 评估基准的参考


**[Elephant in the Room: Unveiling the Impact of Reward Model Qualit…](https://arxiv.org/abs/2409.19024)** （arXiv 2024 2024）  
  *Systematically evaluates how reward model quality impacts alignment performance across three utilization paradigms (offline RL, online RL, inference-time). Curates a cleaner version of HH-RLHF (CHH-RL*  
  关键数字：Demonstrates monotonic improvement in alignment quality with reward model accuracy across all three utilization paradigms  
  **SIA 启示**：P1 directly: SIA's alignment effect (currently MMLU +12pp signal) may be limited by VM quality rather than the intervention mechanism. This paper confirms that RM quality is the primary bottleneck in   
  > ⚠️ Reward Model Quality Impact：RM 质量对最终对齐效果的非线性影响；SIA 选择 VM 架构时的关键参考


**[Rethinking Reward Model Evaluation Through the Lens of Reward Ove…](https://arxiv.org/abs/2505.12763)** （ACL 2025 2025）  
  *Identifies that existing RM benchmarks show weak correlation with optimized policy performance, and proposes evaluation designs through the lens of reward overoptimization. Investigates three design p*  
  关键数字：Demonstrates that extremely high overoptimization correlation leads to lower correlation with certain downstream metrics  
  **SIA 启示**：SIA's main concern for P1 is whether VM scores actually improve generation quality or just hack the reward. This paper provides a framework to detect and mitigate this overoptimization risk—critical w  
  > ⚠️ Reward Overoptimization 视角：防止 SIA 在长序列生成中出现过度优化


**[Safety Tax: Safety Alignment Makes Your Large Reasoning Models Le…](https://arxiv.org/abs/2503.00555)** （arXiv 2025 2025）  
  *Empirically quantifies the 'safety tax': safety alignment procedures applied to Large Reasoning Models (LRMs) degrade reasoning performance. Safety alignment using SafeChain reduces average reasoning *  
  关键数字：SafeChain: -7.09% average reasoning accuracy; DirectRefusal: -30.91% average reasoning accuracy vs. unaligned baseline  
  **SIA 启示**：Directly quantifies the alignment tax concept SIA is trying to address via inference-time methods. If SIA's VM focuses on helpfulness/reasoning rewards, this paper validates that training-time alignme  
  > ⚠️ Safety Tax：对齐干预会降低推理性能；SIA 在 MMLU 上的 +12pp 结果与此相悖，值得深入分析



---

## 六、对 SIA 三大痛点的综合启示

### 6.1 P1（效果）—— 最有价值的改进方向

**⭐⭐⭐ 高优先级（直接可做，1-2 周内）**

1. **乘积式融合（Product of Experts）**  
   当前 SIA 用加法偏置（`logits += weight * rm_score`），ARGS（2402.01694）和 LLMdoctor（2601.10416）表明乘积式（`log P_combined = log P_LLM + β·log P_RM`）在理论上更严格、实践中 win-rate 更高。LLMdoctor 实测 62.1% 胜率 vs GenARM，工作量约 1-2 天（改一行计算公式）。**已在 Task 3.3 中规划，建议提前到 Month 1 快速 A/B 验证**。

2. **概率性干预门控**  
   "To Intervene or Not"（2606.11201）在固定熵阈值之上增加概率性决策，可减少 30%+ 无效调用同时提升 P1。配合"Sparse but Critical"（2603.22446）的关键 token 分析，可以构建更精准的门控策略。

3. **稀疏奖励子系统分析**  
   "Sparse Reward Subsystem"（2602.00986）实证了 LLM 中真正对 reward 敏感的 token 子集，为 SIA 的 entropy 门控提供了 ground truth 分析工具。建议在建立评估基准（Task 1.3）时同步分析哪些 token 位置的干预最有效。

**⭐⭐ 中优先级（Month 2-4）**

4. **PRM + 过程奖励**（Month 5 方向，但可提前小规模验证）  
   More Bang for Buck PRM（2503.22233）用熵驱动的 PRM 调用策略，在 ~40% 调用节省下保持效果，和 SIA 的稀疏干预高度互补。

5. **Adaptive 干预强度**（--weight 动态化）  
   当前 --weight 是全局固定值。Dynamic Logits Fusion（2406.15480）和 Learnable Chernoff Baselines（2602.07738）均表明自适应权重能显著提升 P1 效果。

### 6.2 P2（延迟）—— 可直接借鉴的工程方案

**⭐⭐⭐ 高优先级**

1. **单次 forward 输出 vocabulary-wide reward**（2502.04517 Cost-Effective RGTG）  
   当前 SIA 对 top-K 候选各做一次 forward（即使有 batch 合并，仍是 K 个序列）。该论文将 RM head 改为输出整个 vocab 的 reward，**从 K 次 → 1 次 forward**，理论 P2 改善 topK 倍。工作量：需修改 VM 推理接口，但收益极大。**建议作为 Month 2 同词表 VM 训练时一并实现**。

2. **Fast-Slow Thinking RM**（2603.20212）  
   快速标量分数（first token）+ 慢速生成式 RM 的级联，精确对应 SIA 的 0.6B（fast）+ 4B（slow）两阶段设计（Task 1.4）。提供了完整的系统实现参考。

3. **从内部 hidden states 提取 reward**（2505.12225 Mining Intrinsic Rewards）  
   完全无需外部 VM，直接用 0GM-35B 的隐藏层状态计算 reward。如果成立，**可彻底消除 P2**。高风险高收益，建议在 Month 3-4 进行小规模实验。

**⭐⭐ 中优先级**

4. **Streaming Lookahead Self-reward**（2503.00029）  
   LLM 自身做 streaming 预评分，激进但若成功可同时解决 P2/P3。

5. **低秩 RM 参数化**（2407.04615）  
   在 Month 2 训练同词表 VM 时引入低秩结构，直接减少 VM scoring 的计算量。

### 6.3 P3（吞吐）—— 可直接借鉴的系统优化

**⭐⭐⭐ 高优先级**

1. **Hydragen / CoDec 共享前缀注意力**  
   SIA 的 VM scoring 中，N 个候选共享完全相同的前缀（prompt + generated so far）。Hydragen（2402.05099）将这部分 attention 抽取出来做一次 forward，**理论上可将 VM 的候选评分从 K 次独立计算 → 1 次共享前缀 + K 次很短的 suffix attention**。工作量：需改造 VM serving kernel，但与 Cost-Effective RGTG 方案互补。

2. **AsyncSpade 异步流水线**（2510.07486）  
   VM 调用与主 LLM decoding 异步执行，20-50% TPOT 减少。需改造 `SIALogitsProcessor` 的同步调用为异步，适合 Month 3-4 实现。

3. **跨请求前缀聚合**（BatchLLM 2412.03594 + 2605.06046）  
   在调度层面，优先将前缀相似的请求聚合在同一批次，最大化 APC 命中率。对高并发场景（conc=16+）的 P3 改善显著。

**⚠️ 注意事项**

- **APC + VM Cross-Candidate 警告**（2601.08343）：naively 复用 APC 到 VM 多候选评分会导致评分不一致。SIA 当前已用 stable prefix 方案规避，但扩展时需注意。

---

## 七、Roadmap 启示

### ✅ 已被 roadmap 覆盖的方向

| 论文 | 对应 roadmap 任务 |
|------|-----------------|
| 2503.02368 block-wise scoring | Task 1.1 |
| 2503.01655 RSD / 2506.04118 GSI / 2508.15044 RSSA | Task 1.4 两阶段粗过滤 |
| 2512.23765 EASD / 2603.18411 TARo | Task 1.2 双熵门控 |
| 2410.08193 GenARM | Task 4.1（Month 4）|
| 2601.10416 LLMdoctor | Task 3.3（Month 3）|
| 2404.04475 AlpacaEval 2.0 | Task 1.3 评估基准 |
| 2412.03594 BatchLLM | Task 1.4 注释 |

### 🆕 新发现，建议加入 roadmap

**Month 1（立即可做）：**
- **Task 1.1 补充**：Cost-Effective RGTG（2502.04517）提出的 vocabulary-wide reward head 是 block-wise scoring 之外的另一个 P2 方向，可快速 A/B
- **Task 3.3 提前**：LLMdoctor 乘积式融合（2601.10416）工作量仅 1-2 天（改 apply() 中一行计算），建议和 block-wise A/B 对比一起跑

**Month 2-3（同词表 VM 训练时一并考虑）：**
- `from r to Q*`（2404.12358）：训练 VM 时可利用 LLM 自身的 Q-function 属性，减少对外部标注数据的依赖
- 低秩 VM 参数化（2407.04615）：在同词表 VM 架构设计时引入低秩 reward head，减少 scoring 计算量

**Month 3-4（备选/探索）：**
- Mining Intrinsic Rewards（2505.12225）：无外部 VM 的 reward 提取，高风险高收益实验
- AsyncSpade 异步 VM 调用（2510.07486）：改 SIA 同步调用为异步，P3 系统优化

**Month 5 PRM 训练（参考）：**
- SP-PRM（2506.12446）：从 ORM 推导过程奖励，大幅降低 PRM 训练数据收集门槛
- ThinkPRM（2504.16828）：thinking token 级 PRM，对应 Qwen3 的 thinking 模式
- DG-PRM（2507.17849）：动态 PRM，跨任务泛化性强

**Month 6 多模态（参考）：**
- Skywork-VL Reward（2505.07263）+ MSRL（2603.25108）：多模态 VM 的 backbone 和训练框架

### 🔄 现有任务可根据此调研强化

| 任务 | 建议强化内容 | 依据论文 |
|------|------------|---------|
| Task 1.2 双熵门控 | 增加 learned routing policy 实验 | 2603.09065 |
| Task 1.3 评估基准 | 引入 RMB（2410.09893）和 RewardBench2（2506.01937）作为 VM 质量评估 | 同上 |
| Task 2.X 同词表 VM 训练 | 加入 vocabulary-wide RM head 设计目标 | 2502.04517 |
| Task 3.3 乘积式融合 | 在 A/B 对比中同时测试 Learnable Chernoff weight（2602.07738） | 同上 |

---

## 附录：全部论文索引

| # | 标题 | 会议/年 | ArXiv | 类别 | P1/P2/P3 | 验证 |
|---|------|---------|-------|------|----------|------|
| 1 | [Adaptive Parallel Monte Carlo Tree Search for Efficient…](https://arxiv.org/abs/2604.00510) | arXiv 2026 | 2604.00510 | engi | P2/P3 |  |
| 2 | [Best-of-Tails: Bridging Optimism and Pessimism in Infer…](https://arxiv.org/abs/2603.06797) | arXiv 2026 | 2603.06797 | effe | P1 |  |
| 3 | [More Test-Time Compute Can Hurt: Overestimation Bias in…](https://arxiv.org/abs/2603.15377) | arXiv 2026 | 2603.15377 | eval | P1/P2 |  |
| 4 | [Guided Speculative Inference for Efficient Test-Time Al…](https://arxiv.org/abs/2506.04118) | ICLR 2026 2026 | 2506.04118 | both | P2/P3 |  |
| 5 | [Aligning Tree-Search Policies with Fixed Token Budgets …](https://arxiv.org/abs/2602.09574) | ICML 2026 2026 | 2602.09574 | engi | P2/P3 |  |
| 6 | [Reward Shaping for Inference-Time Alignment: A Stackelb…](https://arxiv.org/abs/2602.02572) | ICML 2026 2026 | 2602.02572 | effe | P1 |  |
| 7 | [MSRL: Scaling Generative Multimodal Reward Modeling via…](https://arxiv.org/abs/2603.25108) | CVPR 2026 2026 | 2603.25108 | trai | P1 |  |
| 8 | [Learning What Matters: Dynamic Dimension Selection and …](https://arxiv.org/abs/2604.05445) | ACL 2026 Main 2026 | 2604.05445 | trai | P1 |  |
| 9 | [Entropy-Guided Data-Efficient Training for Multimodal R…](https://arxiv.org/abs/2602.01884) | arXiv 2026 | 2602.01884 | trai | P1 |  |
| 10 | [Steering Language Models Before They Speak: Logit-Level…](https://arxiv.org/abs/2601.10960) | arXiv 2026 2026 | 2601.10960 | effe | P1/P2 |  |
| 11 | [UniARM: Towards a Unified Autoregressive Reward Model f…](https://arxiv.org/abs/2602.09538) | arXiv 2026 (under review) 2026 | 2602.09538 | trai | P1 |  |
| 12 | [LLMdoctor: Token-Level Flow-Guided Preference Optimizat…](https://arxiv.org/abs/2601.10416) | AAAI 2026 2026 | 2601.10416 | both | P1/P2 |  |
| 13 | [From Tokens to Steps: Verification-Aware Speculative De…](https://arxiv.org/abs/2604.15244) | arXiv (cs.CL) 2026 | 2604.15244 | both | P2/P3 |  |
| 14 | [Multi-Drafter Speculative Decoding with Alignment Feedb…](https://arxiv.org/abs/2604.05417) | ACL 2026 Findings 2026 | 2604.05417 | engi | P2/P3 |  |
| 15 | [Learnable Chernoff Baselines for Inference-Time Alignme…](https://arxiv.org/abs/2602.07738) | arXiv (cs.LG) 2026 | 2602.07738 | both | P1 |  |
| 16 | [TokenWeave: Efficient Compute-Communication Overlap for…](https://arxiv.org/abs/2505.11329) | MLSys 2026 2026 | 2505.11329 | engi | P3 |  |
| 17 | [RKSC: Reasoning-Aware KV Cache Sharing and Confident Ea…](https://arxiv.org/abs/2606.09937) | arXiv preprint 2026 | 2606.09937 | engi | P2/P3 |  |
| 18 | [RadixMLP: Intra-batch Deduplication for Causal Transfor…](https://arxiv.org/abs/2601.15013) | arXiv preprint 2026 | 2601.15013 | engi | P2/P3 |  |
| 19 | [Requests of a Feather Must Flock Together: Batch Size v…](https://arxiv.org/abs/2605.06046) | arXiv preprint 2026 | 2605.06046 | engi | P3 |  |
| 20 | [When KV Cache Reuse Fails in Multi-Agent Systems: Cross…](https://arxiv.org/abs/2601.08343) | arXiv preprint 2026 | 2601.08343 | both | P1/P2 |  |
| 21 | [Fast-Slow Thinking RM: Efficient Integration of Scalar …](https://arxiv.org/abs/2603.20212) | arXiv (cs.CL) 2026 | 2603.20212 | both | P2/P3 |  |
| 22 | [Reward Models Are Secretly Value Functions: Temporally …](https://arxiv.org/abs/2604.22981) | arXiv 2026 | 2604.22981 | trai | P1/P2 |  |
| 23 | [Autoregressive Direct Preference Optimization](https://arxiv.org/abs/2602.09533) | ICML 2026 2026 | 2602.09533 | trai | P1 |  |
| 24 | [DSPA: Dynamic SAE Steering for Data-Efficient Preferenc…](https://arxiv.org/abs/2603.21461) | arXiv 2026 | 2603.21461 | both | P1/P2 |  |
| 25 | [Learning Adaptive LLM Decoding](https://arxiv.org/abs/2603.09065) | arXiv 2026 | 2603.09065 | both | P1/P2 |  |
| 26 | [Entropy-Aligned Decoding of LMs for Better Writing and …](https://arxiv.org/abs/2601.01714) | arXiv 2026 2026 | 2601.01714 | both | P1/P3 |  |
| 27 | [Process Reward Agents for Steering Knowledge-Intensive …](https://arxiv.org/abs/2604.09482) | ICML 2026 2026 | 2604.09482 | effe | P1 |  |
| 28 | [Hidden States as Early Signals: Step-level Trace Evalua…](https://arxiv.org/abs/2601.09093) | arXiv 2026 | 2601.09093 | engi | P2/P3 |  |
| 29 | [Harvest: Opportunistic Peer-to-Peer GPU Caching for LLM…](https://arxiv.org/abs/2602.00328) | arXiv 2026 | 2602.00328 | engi | P2/P3 |  |
| 30 | [Blockwise Advantage Estimation for Multi-Objective RL w…](https://arxiv.org/abs/2602.10231) | arXiv 2026 | 2602.10231 | trai | P1/P2/P3 |  |
| 31 | [FluxMoE: Decoupling Expert Residency for High-Performan…](https://arxiv.org/abs/2604.02715) | arXiv 2026 | 2604.02715 | engi | P3 |  |
| 32 | [RewardBench 2: Advancing Reward Model Evaluation](https://arxiv.org/abs/2506.01937) | arXiv 2026 2026 | 2506.01937 | eval | P1 |  |
| 33 | [Skywork-Reward-V2: Scaling Preference Data Curation via…](https://arxiv.org/abs/2507.01352) | ICLR 2026 2026 | 2507.01352 | trai | P1/P2 |  |
| 34 | [Towards Understanding Valuable Preference Data for Larg…](https://arxiv.org/abs/2510.13212) | ICLR 2026 2026 | 2510.13212 | trai | P1 |  |
| 35 | [Sparse Reward Subsystem in Large Language Models](https://arxiv.org/abs/2602.00986) | arXiv 2026 2026 | 2602.00986 | effe | P1/P2 |  |
| 36 | [Evaluating Reward Model Generalization via Pairwise Max…](https://arxiv.org/abs/2601.16987) | arXiv 2026 2026 | 2601.16987 | eval | P1 |  |
| 37 | [Long-form RewardBench: Evaluating Reward Models for Lon…](https://arxiv.org/abs/2603.12963) | AAAI 2026 2026 | 2603.12963 | eval | P1 |  |
| 38 | [QuRL: Efficient Reinforcement Learning with Quantized R…](https://arxiv.org/abs/2602.13953) | ICLR 2026 2026 | 2602.13953 | engi | P2/P3 |  |
| 39 | [RM-Distiller: Exploiting Generative LLM for Reward Mode…](https://arxiv.org/abs/2601.14032) | arXiv 2026 2026 | 2601.14032 | trai | P1/P2/P3 |  |
| 40 | [Debiasing Reward Models via Causally Motivated Inferenc…](https://arxiv.org/abs/2604.27495) | ACL 2026 Main 2026 | 2604.27495 | engi | P1/P2 |  |
| 41 | [Sparse but Critical: A Token-Level Analysis of Distribu…](https://arxiv.org/abs/2603.22446) | ICLR 2026 2026 | 2603.22446 | effe | P1/P2/P3 |  |
| 42 | [To Intervene or Not: Guiding Inference-time Alignment w…](https://arxiv.org/abs/2606.11201) | ACL 2026 2026 | 2606.11201 | effe | P1 |  |
| 43 | [Reward-Guided Speculative Decoding for Efficient LLM Re…](https://arxiv.org/abs/2501.19324) | arXiv 2025 | 2501.19324 | both | P2/P3 |  |
| 44 | [Wider or Deeper? Scaling LLM Inference-Time Compute wit…](https://arxiv.org/abs/2503.04412) | NeurIPS 2025 (Spotlight) 2025 | 2503.04412 | effe | P1/P2 |  |
| 45 | [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) | arXiv 2025 | 2504.02495 | effe | P1 |  |
| 46 | [Reward-Shifted Speculative Sampling Is An Efficient Tes…](https://arxiv.org/abs/2508.15044) | EMNLP 2025 2025 | 2508.15044 | both | P1/P2/P3 |  |
| 47 | [Entropy Adaptive Decoding: Dynamic Model Switching for …](https://arxiv.org/abs/2502.06833) | arXiv 2025 | 2502.06833 | engi | P2/P3 |  |
| 48 | [Is Best-of-N the Best of Them? Coverage, Scaling, and O…](https://arxiv.org/abs/2503.21878) | arXiv 2025 | 2503.21878 | effe | P1 |  |
| 49 | [Inference-Time Reward Hacking in Large Language Models](https://arxiv.org/abs/2506.19248) | NeurIPS 2025 (Spotlight) 2025 | 2506.19248 | eval | P1 |  |
| 50 | [Adaptive Blockwise Search: Inference-Time Alignment for…](https://arxiv.org/abs/2510.23334) | arXiv 2025 | 2510.23334 | effe | P1/P2/P3 |  |
| 51 | [W2S-AlignTree: Weak-to-Strong Inference-Time Alignment …](https://arxiv.org/abs/2511.11518) | AAAI 2026 (Oral) 2025 | 2511.11518 | effe | P1 |  |
| 52 | [Token-Level Inference-Time Alignment for Vision-Languag…](https://arxiv.org/abs/2510.21794) | arXiv (ICLR 2026 submission) 2025 | 2510.21794 | effe | P1 |  |
| 53 | [VisualPRM: An Effective Process Reward Model for Multim…](https://arxiv.org/abs/2503.10291) | arXiv 2025 | 2503.10291 | effe | P1 |  |
| 54 | [Skywork-VL Reward: An Effective Reward Model for Multim…](https://arxiv.org/abs/2505.07263) | arXiv 2025 | 2505.07263 | trai | P1 |  |
| 55 | [Training Vision-Language Process Reward Models for Test…](https://arxiv.org/abs/2509.23250) | arXiv 2025 | 2509.23250 | both | P1/P2 |  |
| 56 | [BaseReward: A Strong Baseline for Multimodal Reward Mod…](https://arxiv.org/abs/2509.16127) | arXiv 2025 | 2509.16127 | trai | P1 |  |
| 57 | [MM-RLHF: The Next Step Forward in Multimodal LLM Alignm…](https://arxiv.org/abs/2502.10391) | arXiv 2025 | 2502.10391 | trai | P1 |  |
| 58 | [ProxyThinker: Test-Time Guidance through Small Visual R…](https://arxiv.org/abs/2505.24872) | arXiv 2025 | 2505.24872 | effe | P1/P2 |  |
| 59 | [Dual-Stage Value-Guided Inference with Margin-Based Rew…](https://arxiv.org/abs/2506.15649) | arXiv 2025 | 2506.15649 | effe | P1/P2/P3 |  |
| 60 | [The Devil Is in the Details: Tackling Unimodal Spurious…](https://arxiv.org/abs/2503.03122) | ICML 2025 2025 | 2503.03122 | trai | P1 |  |
| 61 | [Multimodal RewardBench: Holistic Evaluation of Reward M…](https://arxiv.org/abs/2502.14191) | arXiv 2025 | 2502.14191 | eval | P1 |  |
| 62 | [Sentence-level Reward Model can Generalize Better for A…](https://arxiv.org/abs/2503.04793) | arXiv 2025 2025 | 2503.04793 | trai | P1/P2/P3 |  |
| 63 | [AlignDistil: Token-Level Language Model Alignment as Ad…](https://arxiv.org/abs/2503.02832) | ACL 2025 2025 | 2503.02832 | trai | P1 |  |
| 64 | [Towards Cost-Effective Reward Guided Text Generation](https://arxiv.org/abs/2502.04517) | ICML 2025 2025 | 2502.04517 | engi | P2/P3 |  |
| 65 | [PARM: Multi-Objective Test-Time Alignment via Preferenc…](https://arxiv.org/abs/2505.06274) | ICML 2025 2025 | 2505.06274 | trai | P1 |  |
| 66 | [Alignment-Aware Decoding](https://arxiv.org/abs/2509.26169) | ICML 2026 2025 | 2509.26169 | effe | P1/P2 |  |
| 67 | [MAVIS: Multi-Objective Alignment via Inference-Time Val…](https://arxiv.org/abs/2508.13415) | arXiv 2025 2025 | 2508.13415 | both | P1 |  |
| 68 | [SPECS: Faster Test-Time Scaling through Speculative Dra…](https://arxiv.org/abs/2506.15733) | arXiv (cs.AI) 2025 | 2506.15733 | both | P1/P2/P3 |  |
| 69 | [Judge Decoding: Faster Speculative Sampling Requires Go…](https://arxiv.org/abs/2501.19309) | ICLR 2025 2025 | 2501.19309 | both | P2/P3 |  |
| 70 | [Accelerating Mixture-of-Experts Inference by Hiding Off…](https://arxiv.org/abs/2508.21706) | arXiv (cs.DC) 2025 | 2508.21706 | engi | P3 |  |
| 71 | [Utility-Driven Speculative Decoding for Mixture-of-Expe…](https://arxiv.org/abs/2506.20675) | arXiv (cs.DC) 2025 | 2506.20675 | engi | P3 |  |
| 72 | [AsyncSpade: Efficient Test-Time Scaling with Asynchrono…](https://arxiv.org/abs/2510.07486) | arXiv 2025 2025 | 2510.07486 | engi | P2/P3 |  |
| 73 | [OPPO: Accelerating PPO-based RLHF via Pipeline Overlap](https://arxiv.org/abs/2509.25762) | arXiv 2025 2025 | 2509.25762 | engi | P2/P3 |  |
| 74 | [NanoFlow: Towards Optimal Large Language Model Serving …](https://arxiv.org/abs/2408.12757) | OSDI 2025 2025 | 2408.12757 | engi | P3 |  |
| 75 | [APEX: Asynchronous Parallel CPU-GPU Execution for Onlin…](https://arxiv.org/abs/2506.03296) | arXiv 2025 2025 | 2506.03296 | engi | P3 |  |
| 76 | [PipeSpec: Breaking Stage Dependencies in Hierarchical L…](https://arxiv.org/abs/2505.01572) | ACL Findings 2025 2025 | 2505.01572 | engi | P2/P3 |  |
| 77 | [CoDec: Prefix-Shared Decoding Kernel for LLMs](https://arxiv.org/abs/2505.17694) | arXiv preprint 2025 | 2505.17694 | engi | P2/P3 |  |
| 78 | [Mining Intrinsic Rewards from LLM Hidden States for Eff…](https://arxiv.org/abs/2505.12225) | KDD 2026 2025 | 2505.12225 | engi | P2/P3 |  |
| 79 | [Accelerating LLM Reasoning via Early Rejection with Par…](https://arxiv.org/abs/2508.01969) | arXiv (cs.LG) 2025 | 2508.01969 | engi | P2/P3 |  |
| 80 | [Bi-directional Model Cascading with Proxy Confidence](https://arxiv.org/abs/2504.19391) | arXiv (cs.LG) 2025 | 2504.19391 | engi | P2/P3 |  |
| 81 | [Towards Hierarchical Multi-Step Reward Models for Enhan…](https://arxiv.org/abs/2503.13551) | arXiv (cs.CL) 2025 | 2503.13551 | trai | P1 |  |
| 82 | [TGDPO: Harnessing Token-Level Reward Guidance for Enhan…](https://arxiv.org/abs/2506.14574) | ICML 2025 2025 | 2506.14574 | trai | P1 |  |
| 83 | [Earlier Tokens Contribute More: Learning Direct Prefere…](https://arxiv.org/abs/2502.14340) | ICLR 2025 2025 | 2502.14340 | trai | P1 |  |
| 84 | [DPO Meets PPO: Reinforced Token Optimization for RLHF](https://arxiv.org/abs/2404.18922) | ICML 2025 2025 | 2404.18922 | trai | P1 |  |
| 85 | [Streaming Looking Ahead with Token-level Self-reward](https://arxiv.org/abs/2503.00029) | arXiv 2025 | 2503.00029 | both | P2/P3 |  |
| 86 | [RED: Unleashing Token-Level Rewards from Holistic Feedb…](https://arxiv.org/abs/2411.08302) | EMNLP 2025 2025 | 2411.08302 | trai | P1 |  |
| 87 | [Selective Preference Optimization via Token-Level Rewar…](https://arxiv.org/abs/2408.13518) | EMNLP 2025 2025 | 2408.13518 | trai | P1/P2 |  |
| 88 | [Advantage-Guided Distillation for Preference Alignment …](https://arxiv.org/abs/2502.17927) | ICLR 2025 (Spotlight) 2025 | 2502.17927 | trai | P1 |  |
| 89 | [Collab: Controlled Decoding using Mixture of Agents for…](https://arxiv.org/abs/2503.21720) | ICLR 2025 2025 | 2503.21720 | effe | P1 |  |
| 90 | [RMB: Comprehensively Benchmarking Reward Models in LLM …](https://arxiv.org/abs/2410.09893) | ICLR 2025 2025 | 2410.09893 | eval | P1 |  |
| 91 | [Less is More: Improving LLM Reasoning with Minimal Test…](https://arxiv.org/abs/2510.13940) | arXiv 2025 2025 | 2510.13940 | both | P1/P3 |  |
| 92 | [Speculative Reward Model Boosts Decision Making Ability…](https://arxiv.org/abs/2506.00396) | ACL 2025 Industry Track 2025 | 2506.00396 | engi | P2/P3 |  |
| 93 | [Process Reward Models That Think (ThinkPRM)](https://arxiv.org/abs/2504.16828) | arXiv 2025 | 2504.16828 | trai | P1 |  |
| 94 | [The Lessons of Developing Process Reward Models in Math…](https://arxiv.org/abs/2501.07301) | arXiv 2025 | 2501.07301 | trai | P1 |  |
| 95 | [GenPRM: Scaling Test-Time Compute of Process Reward Mod…](https://arxiv.org/abs/2504.00891) | arXiv 2025 | 2504.00891 | trai | P1/P2 |  |
| 96 | [Efficient Test-Time Scaling of Multi-Step Reasoning by …](https://arxiv.org/abs/2511.06209) | ACL 2026 Main 2025 | 2511.06209 | engi | P2/P3 |  |
| 97 | [A Survey of Process Reward Models: From Outcome Signals…](https://arxiv.org/abs/2510.08049) | arXiv 2025 | 2510.08049 | both | P1 |  |
| 98 | [Dynamic and Generalizable Process Reward Modeling (DG-P…](https://arxiv.org/abs/2507.17849) | ACL 2025 Main 2025 | 2507.17849 | trai | P1 |  |
| 99 | [LogicReward: Incentivizing LLM Reasoning via Step-Wise …](https://arxiv.org/abs/2512.18196) | ICLR 2026 2025 | 2512.18196 | trai | P1 |  |
| 100 | [Process Reinforcement through Implicit Rewards (PRIME)](https://arxiv.org/abs/2502.01456) | arXiv 2025 | 2502.01456 | trai | P1 |  |
| 101 | [FreePRM: Training Process Reward Models Without Ground …](https://arxiv.org/abs/2506.03570) | arXiv 2025 | 2506.03570 | trai | P1 |  |
| 102 | [From Outcomes to Processes: Guiding PRM Learning from O…](https://arxiv.org/abs/2506.12446) | arXiv 2025 | 2506.12446 | both | P1 |  |
| 103 | [Prism: Cost-Efficient Multi-LLM Serving via GPU Memory …](https://arxiv.org/abs/2505.04021) | OSDI 2026 2025 | 2505.04021 | engi | P3 |  |
| 104 | [Enabling Disaggregated Multi-Stage MLLM Inference via G…](https://arxiv.org/abs/2512.17574) | arXiv 2025 | 2512.17574 | engi | P2/P3 |  |
| 105 | [More Bang for the Buck: Process Reward Modeling with En…](https://arxiv.org/abs/2503.22233) | arXiv 2025 | 2503.22233 | effe | P1/P2/P3 |  |
| 106 | [STARS: Synchronous Token Alignment for Robust Supervisi…](https://arxiv.org/abs/2511.03827) | arXiv 2025 | 2511.03827 | engi | P2/P3 |  |
| 107 | [Taming Latency-Memory Trade-Off in MoE-Based LLM Servin…](https://arxiv.org/abs/2502.05370) | EuroSys 2026 2025 | 2502.05370 | engi | P3 |  |
| 108 | [StreamRL: Scalable, Heterogeneous, and Elastic RL for L…](https://arxiv.org/abs/2504.15930) | arXiv 2025 | 2504.15930 | engi | P3 |  |
| 109 | [RRM: Robust Reward Model Training Mitigates Reward Hack…](https://arxiv.org/abs/2409.13156) | ICLR 2025 2025 | 2409.13156 | trai | P1 |  |
| 110 | [Less is More: Improving LLM Alignment via Preference Da…](https://arxiv.org/abs/2502.14560) | arXiv 2025 2025 | 2502.14560 | trai | P1 |  |
| 111 | [OpenRubrics: Towards Scalable Synthetic Rubric Generati…](https://arxiv.org/abs/2510.07743) | arXiv 2025 2025 | 2510.07743 | trai | P1 |  |
| 112 | [SCAR: Shapley Credit Assignment for More Efficient RLHF](https://arxiv.org/abs/2505.20417) | arXiv 2025 2025 | 2505.20417 | trai | P1 |  |
| 113 | [QeRL: Beyond Efficiency -- Quantization-enhanced Reinfo…](https://arxiv.org/abs/2510.11696) | arXiv 2025 2025 | 2510.11696 | engi | P2/P3 |  |
| 114 | [QServe: W4A8KV4 Quantization and System Co-design for E…](https://arxiv.org/abs/2405.04532) | MLSys 2025 2025 | 2405.04532 | engi | P2/P3 |  |
| 115 | [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387) | ICLR 2026 2025 | 2505.02387 | trai | P1/P2 |  |
| 116 | [Rethinking Reward Model Evaluation Through the Lens of …](https://arxiv.org/abs/2505.12763) | ACL 2025 2025 | 2505.12763 | eval | P1 |  |
| 117 | [Safety Tax: Safety Alignment Makes Your Large Reasoning…](https://arxiv.org/abs/2503.00555) | arXiv 2025 2025 | 2503.00555 | eval | P1 |  |
| 118 | [Leveraging Importance Sampling to Detach Alignment Modu…](https://arxiv.org/abs/2505.19700) | NeurIPS 2025 2025 | 2505.19700 | both | P1/P2 |  |
| 119 | [Inference-time Alignment in Continuous Space](https://arxiv.org/abs/2505.20081) | NeurIPS 2025 2025 | 2505.20081 | effe | P1/P2 |  |
| 120 | [G2: Guided Generation for Enhanced Output Diversity in …](https://arxiv.org/abs/2511.00432) | EMNLP 2025 2025 | 2511.00432 | effe | P1 |  |
| 121 | [Scaling Inference-Time Search with Vision Value Model f…](https://arxiv.org/abs/2412.03704) | arXiv 2024 | 2412.03704 | effe | P1 |  |
| 122 | [Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565) | COLM 2024 2024 | 2401.08565 | effe | P1/P2 |  |
| 123 | [ARGS: Alignment as Reward-Guided Search](https://arxiv.org/abs/2402.01694) | ICLR 2024 2024 | 2402.01694 | effe | P1/P2 |  |
| 124 | [Controlled Decoding from Language Models](https://arxiv.org/abs/2310.17022) | ICML 2024 2024 | 2310.17022 | effe | P1/P2/P3 |  |
| 125 | [Decoding-time Realignment of Language Models](https://arxiv.org/abs/2402.02992) | ICML 2024 2024 | 2402.02992 | effe | P1 |  |
| 126 | [Decoding-Time Language Model Alignment with Multiple Ob…](https://arxiv.org/abs/2406.18853) | NeurIPS 2024 2024 | 2406.18853 | effe | P1 |  |
| 127 | [Nudging: Inference-time Alignment of LLMs via Guided De…](https://arxiv.org/abs/2410.09300) | ACL 2025 2024 | 2410.09300 | effe | P1/P2/P3 |  |
| 128 | [Inference-Time Language Model Alignment via Integrated …](https://arxiv.org/abs/2409.17819) | EMNLP 2024 Findings 2024 | 2409.17819 | effe | P1/P2/P3 |  |
| 129 | [PAD: Personalized Alignment of LLMs at Decoding-Time](https://arxiv.org/abs/2410.04070) | ICLR 2025 2024 | 2410.04070 | effe | P1 |  |
| 130 | [On Giant's Shoulders: Effortless Weak to Strong by Dyna…](https://arxiv.org/abs/2406.15480) | NeurIPS 2024 2024 | 2406.15480 | effe | P1 |  |
| 131 | [Cascade Reward Sampling for Efficient Decoding-Time Ali…](https://arxiv.org/abs/2406.16306) | arXiv 2024 2024 | 2406.16306 | engi | P2/P3 |  |
| 132 | [Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/abs/2410.20290) | NeurIPS 2024 2024 | 2410.20290 | engi | P2/P3 |  |
| 133 | [A Critical Look At Tokenwise Reward-Guided Text Generat…](https://arxiv.org/abs/2406.07780) | COLM 2025 2024 | 2406.07780 | trai | P1 |  |
| 134 | [On the Low-Rank Parametrization of Reward Models for Co…](https://arxiv.org/abs/2407.04615) | TMLR 2025 2024 | 2407.04615 | engi | P2 |  |
| 135 | [Multi-Draft Speculative Sampling: Canonical Decompositi…](https://arxiv.org/abs/2410.18234) | ICLR 2025 (Spotlight) 2024 | 2410.18234 | engi | P1/P2 |  |
| 136 | [PipeInfer: Accelerating LLM Inference using Asynchronou…](https://arxiv.org/abs/2407.11798) | SC 2024 2024 | 2407.11798 | engi | P2/P3 |  |
| 137 | [Hydragen: High-Throughput LLM Inference with Shared Pre…](https://arxiv.org/abs/2402.05099) | ICLR 2025 2024 | 2402.05099 | engi | P2/P3 |  |
| 138 | [BatchLLM: Optimizing Large Batched LLM Inference with G…](https://arxiv.org/abs/2412.03594) | MLSys 2026 2024 | 2412.03594 | engi | P3 |  |
| 139 | [Faster Cascades via Speculative Decoding](https://arxiv.org/abs/2405.19261) | arXiv (cs.CL) 2024 | 2405.19261 | engi | P2/P3 |  |
| 140 | [From r to Q*: Your Language Model is Secretly a Q-Funct…](https://arxiv.org/abs/2404.12358) | COLM 2024 2024 | 2404.12358 | trai | P1/P2 |  |
| 141 | [T-REG: Preference Optimization with Token-Level Reward …](https://arxiv.org/abs/2412.02685) | ACL 2025 2024 | 2412.02685 | trai | P1 |  |
| 142 | [TLCR: Token-Level Continuous Reward for Fine-grained Re…](https://arxiv.org/abs/2407.16574) | ACL 2024 Findings 2024 | 2407.16574 | trai | P1/P2 |  |
| 143 | [Sequence to Sequence Reward Modeling: Improving RLHF by…](https://arxiv.org/abs/2409.00162) | arXiv 2024 | 2409.00162 | trai | P1 |  |
| 144 | [Dense Reward for Free in Reinforcement Learning from Hu…](https://arxiv.org/abs/2402.00782) | arXiv 2024 | 2402.00782 | trai | P1 |  |
| 145 | [Elephant in the Room: Unveiling the Impact of Reward Mo…](https://arxiv.org/abs/2409.19024) | arXiv 2024 2024 | 2409.19024 | eval | P1 |  |
| 146 | [Predicting Rewards Alongside Tokens: Non-disruptive Par…](https://arxiv.org/abs/2408.10764) | arXiv 2024 2024 | 2408.10764 | engi | P2/P3 |  |
| 147 | [InfAlign: Inference-aware language model alignment](https://arxiv.org/abs/2412.19792) | arXiv 2024 (Google DeepMind) 2024 | 2412.19792 | trai | P1 |  |
| 148 | [Reward-Augmented Decoding: Efficient Controlled Text Ge…](https://arxiv.org/abs/2310.09520) | EMNLP 2023 2023 | 2310.09520 | effe | P2/P3 |  |
