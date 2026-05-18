# SIA AlpacaEval 样本分析：SIA 在哪里答得更好？

## 分析目标

AlpacaEval 实验已经证明 SIA 将平均 reward 从 12.29 提升至 13.92（+13.2%）。本文档深入分析：**具体是哪类问题从 SIA 干预中受益最多，以及 Value Model 在什么情况下将模型从错误路径拉了回来。**

评分使用独立的 Skywork-Reward-V2-Llama-3.1-8B，与引导用的 Qwen3-4B + VM LoRA 完全不同，排除自评自偏差。

---

## 一、整体分布

| 指标 | 值 |
|------|-----|
| 总样本数 | 805 |
| SIA > Base（提升） | **478 题（59.4%）** |
| SIA < Base（下降） | 312 题（38.8%） |
| SIA = Base（持平） | 15 题 |
| Base 平均 reward | 12.29 |
| SIA 平均 reward | **13.92** |

超过半数（59.4%）的题目 SIA 优于 Base，下降样本数量更少，且下降幅度平均也更小（后文分类统计可见）。

---

## 二、提升最集中的场景

### 场景 1：阻止事实性幻觉（最显著的提升来源）

这类提升最为极端：Base 给出了错误信息，SIA 将其纠正。以下三个样本最为典型：

**① 知识问答中的幻觉（id=72）**

- 问题：*Who was the first lead vocal for the rock band Starship?*
- Base（reward=-15.94）：给出 Jeanne Pownall / Shawn Colvin，**两个名字均为编造**，且配有完全虚构的专辑和歌曲名
- SIA（reward=+3.64）：给出 Grace Slick，基本正确，未出现虚构内容

**② 角色扮演中的身份错误（id=390）**

- 问题：*Hello there Obi One Kenobi*（向 Obi-Wan Kenobi 打招呼）
- Base（reward=-10.12）：以 Darth Vader / Anakin Skywalker 的身份回应，完全答非所问
- SIA（reward=+12.19）：以 Obi-Wan Kenobi 身份回应，并顺带指出名字拼写问题（"Obi One" → "Obi-Wan"）

**③ 细节事实错误（id=116）**

- 问题：请给我一道拉脱维亚肉丸汤（Frikadeļu zupa）的食谱
- Base（reward=-8.12）：食谱内容基本正确，但将这道菜错误标注为 **Lithuanian（立陶宛）菜**
- SIA（reward=+8.62）：正确标注为 **Latvian（拉脱维亚）菜**

这 12 个"从负分拉到正分"的样本（Base < -5 且 SIA > 0），Base 平均 reward 约 -10.7，SIA 平均约 +9.9，单题平均提升约 20 分，是全部样本中贡献最集中的部分。

---

### 场景 2：准确理解问题意图

Base 在语义理解上出现偏差时，SIA 的干预能让模型抓住问题的真实含义。

**典型样本（id=600）**

- 问题：*Give a grammar tip on how to end a German verb.*
- Base（reward=-9.62）：将"end a German verb"理解为"动词置于句末"，回答了从句中动词后置的语序规则（V-final in subordinate clauses）。这是一个语法知识点，但**不是问题在问的**。
- SIA（reward=+12.75）：将"end a German verb"理解为"动词的变位词尾"，回答了现在时各人称变位规则（-e, -st, -t, -en, -t, -en），**正是提问者需要的答案**。

```
Base 回答开头：
  "Tip: Place the verb at the end in subordinate clauses.
   In subordinate clauses (e.g., after weil, obwohl), the main verb
   goes to the end of the clause..."

SIA 回答开头：
  "A key grammar tip on how to end a German verb (i.e., the verb
   conjugation) is to understand that German verbs change their
   endings depending on the subject and the tense used..."
```

---

### 场景 3：情感支持类问题的针对性

Value Model 训练目标包含 Helpfulness，对"有没有真正回答用户的问题"非常敏感。

**典型样本（id=385）—— 全部 805 题中提升最大（+29.06）**

- 问题：*I have a project due in 2 weeks that will determine if I get my diploma. I'm nowhere near the end of it. I just spent an entire afternoon labelling prompts on Open Assistant. Am I going in the right direction?*
- Base（reward=-7.19）：开头共情语言后，绕到"如果这是你项目的一部分则在正确方向，否则……"——**没有给出明确判断**，反而给了一堆泛泛的时间管理建议
- SIA（reward=+21.88）：明确接住了"Am I going in the right direction?"这个核心问题，**直接判断"labelling prompts 不能帮你完成学位论文"**，并指出这很可能是一次分心，建议立即聚焦于项目本身

```
SIA 回答关键段落：
  "Regarding your question: Are you going in the right direction?
   Well, spending an entire afternoon labeling prompts for Open
   Assistant might not directly help you finish your project.
   Unless your project is specifically about training or fine-tuning
   language models... it seems like time spent on that might be a
   detour."
```

---

### 场景 4：解释/概念类问题的结构清晰度

这类问题（解释、定义、对比概念）是提升最稳定的类别之一，Base 答案通常正确但组织松散，SIA 倾向于产出结构更清晰、层次更分明的回答。

**典型样本（id=212）**

- 问题：*Is queue an ADT or a data structure?*
- Base（reward=+15.44）和 SIA（reward=+35.75）都给出了正确结论（两者都是），但 SIA 的回答：先精确定义 ADT 的抽象性，再解释具体实现层的数据结构，概念边界清晰；Base 的回答同样正确，但对两个概念的区分稍显模糊。

---

## 三、各类题目统计

| 题目类型 | 提升样本数 | 平均提升 | 下降样本数 | 平均下降 |
|----------|-----------|---------|-----------|---------|
| 解释/概念 | 70 | **+6.26** | 44 | -4.20 |
| 建议/指导 | 24 | +4.64 | 14 | -6.30 |
| 写作/创意 | 74 | +4.42 | 41 | -3.00 |
| 编程/技术 | 20 | +4.77 | 12 | -3.57 |
| 列举/清单 | 23 | +5.26 | 26 | -3.93 |
| 食谱/烹饪 | 72 | +4.81 | 57 | -4.01 |
| 其他 | 188 | +5.33 | 107 | -3.42 |

"解释/概念"类平均提升最高（+6.26），且提升/下降比例最优（70:44 ≈ 1.6:1）。"建议/指导"类提升/下降比例最优（24:14 ≈ 1.7:1）。

"食谱/烹饪"类样本量最大，下降样本也最多（72:57），说明 Value Model 在烹饪细节层面并不总是占优——这类任务对专业知识要求高，SIA 引导方向正确但细节不总更好。

---

## 四、结论

**SIA 提升最集中的场景是：Value Model 在高熵决策点介入，阻止了模型走上一条质量明显更低的路径。** 具体体现在：

1. **防止幻觉**：当模型准备输出编造事实时，Value Model 给高质量 token 更高分数，将生成轨迹拉向真实内容。这是提升幅度最大的来源（单题可提升 15–29 分）。

2. **准确理解意图**：对于有歧义的问题，Value Model 倾向于给"真正回答用户所问"的方向更高分，而非"技术上相关但答非所问"的方向。

3. **情感与 Helpfulness**：Value Model 在 Helpfulness 数据上训练，能识别"是否真正帮到用户"，对给出明确、直接答案的 token 给予更高引导权重。

4. **SIA 并非万能**：下降的 312 题中，Value Model 有时会将模型推向过于谨慎或结构过于冗长的回答，这与 Helpfulness 数据的偏向有关。总体而言，提升样本的平均提升幅度（约 +5）显著大于下降样本的平均下降幅度（约 -3.9），净效果为正。
