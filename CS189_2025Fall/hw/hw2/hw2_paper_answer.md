# Homework 2 Paper: Ranking LLM

CS 189/289A | Fall 2025

**Due:** Friday, October 17th at 11:59 pm

**Deliverables.** Submit a PDF of your write-up to Gradescope HW2 Paper

## Overview

Large Language Models (LLMs) are capable of doing amazing things! But how do we evaluate their performance? Unlike standard classification tasks, evaluation of LLM outputs involves nuance, style, and human judgment. This homework will help you explore two recent approaches to evaluation: Chatbot Arena and VibeCheck.

You are encouraged to skim related works, but focus on the problems, current solutions, contributions, methods, and limitations.

This assignment is not about memorizing details, but about developing the ability to read research papers methodically. Most research papers follow a common structure: they first motivate a **problem**, then describe **current solutions and their limitations**, followed by the **proposed solution and key insights**, the **methods** used, and finally a discussion of **limitations**. This homework is designed to help you practice and internalize this process of critical reading as you answer the questions. We have also provided pointers to relevant section that you might want to pay more attention to for each question.

- Paper: [Chatbot Arena (arXiv:2403.04132)](https://arxiv.org/abs/2403.04132), [VibeCheck (arXiv:2410.12851)](https://arxiv.org/abs/2410.12851)
- Platform: [VibeCheck](https://bench-mark.org/)

For each question, we are looking for high-level concise answers, no need to write essays.

## Paper Questions

### Chatbot Arena

#### Q1. [Problem]

What problem is Chatbot Arena aiming to solve? Why is evaluating generative models challenging compared to classification tasks? [Section: 1. Introduction]

**Answer:**

Chatbot Arena 旨在解决现有基准难以充分评估大语言模型在真实、开放式任务中是否符合用户偏好的问题。分类任务通常有明确的正确标签，可以用准确率等指标评估；生成任务可能有多个合理答案，回答质量还涉及有用性、风格和主观偏好，因此难以仅通过标准答案衡量。[参考：Introduction](https://arxiv.org/html/2403.04132v1#S1)

#### Q2. [Current works]

What are current approaches to solving this problem and why are they not sufficient? [Section: Related Works, LLM Benchmark]

**Hint:** What are the categories of LLM benchmarks outlined in the paper, and what are the limitations of each category?

**Answer:**

论文按问题来源（静态或实时更新）和评估依据（标准答案或人类偏好）将基准分为四类：

- 静态问题 + 标准答案：通常难以反映开放式、交互式使用，固定测试集还可能受到训练数据污染；复杂任务也未必存在唯一标准答案。
- 静态问题 + 人类偏好：可以评价开放式回答，但固定问题集仍有污染和覆盖范围有限的问题；使用 LLM 代替人类评分时，也只能近似人类偏好。
- 实时问题 + 标准答案：新题可以减少污染风险，但仍受限于能够确定正确答案的任务。
- 实时问题 + 人类偏好：更接近真实交互，但已有相关工作通常局限于特定组织，缺少开放、大规模的评测平台。

[参考：Introduction 与 Related Work](https://arxiv.org/html/2403.04132v1#S2)

#### Q3. [Proposed solution]

What are the inputs and outputs of Chatbot Arena?

**Hint:** outputs exist both per-battle and aggregated.

**Answer:**

交互输入是用户的问题或多轮提示，以及参与比较的两个匿名模型。模型生成回答后，用户提交偏好投票；这些回答和投票构成后续统计评测的数据。

单场输出是一条两两比较记录，包括模型回答及用户选择的结果：模型 A 更好、模型 B 更好、平局或两者都不好。汇总输出是根据大量投票估计的模型评分、排名及置信区间。[参考：Sections 3–5](https://arxiv.org/html/2403.04132v1#S3)

#### Q4. [Key insight / contributions]

Why does the Chatbot Arena evaluation approach address the issues of prior benchmarks? What are their main contributions? (These could be methods, software artifacts, formalization of a problem, datasets, etc.)

**Answer:**

Arena 通过众包收集多样化、持续更新的真实用户问题，减少对固定测试集的依赖；通过匿名回答的两两比较直接衡量人类偏好，能够评价缺少唯一标准答案的开放式任务。

主要贡献包括：构建开放、大规模、基于实时用户交互的众包评测平台；分析提示词的多样性和质量、投票质量及人类反馈；提供超过 10 万组两两比较投票的人类偏好数据集；设计主动选择模型对的高效采样算法，提高评分和排名估计的样本效率。[参考：Introduction](https://arxiv.org/html/2403.04132v1#S1)

#### Q5. [Method details]

**(a)** What is a Bradley–Terry coefficient, and how is this used in the Chatbot Arena scoring?

**Answer (a):**

Bradley–Terry（BT）系数是表示模型相对实力的参数。设模型 $i$ 和 $j$ 的系数为 $\beta_i$ 和 $\beta_j$，则模型 $i$ 获胜的概率为：

$$
P(i \text{ beats } j)=\frac{e^{\beta_i}}{e^{\beta_i}+e^{\beta_j}}.
$$

Arena 根据用户的两两比较投票，通过最大似然估计（等价于最小化二元交叉熵）拟合这些系数，并据此评分和排名。系数越大，预测的获胜概率越高；只有系数之间的差值有意义。[参考：Section 4](https://arxiv.org/html/2403.04132v1#S4)

**(b)** Section 4 of the Chatbot Arena paper mentions that the model scoring system of the original Chatbot Arena interface used Elo scores instead of Bradley–Terry coefficients because they are “better for the purpose of statistical estimation.” Why are Bradley–Terry coefficients more appropriate for this setting?

**Hint:** Elo is an online metric, meaning that the model scores are updated after every battle, while Bradley–Terry is offline—it fits all match outcomes at once to find coefficients that best explain the entire dataset. Think about what assumptions Elo makes about the models.

**Additional resource:** “As a starting point, we show that the Elo score provably fails to extract the transitive component of some elementary transitive games.” ([Bertrand et al., 2023](https://arxiv.org/pdf/2206.12301))

**Answer (b):**

Elo 在每场对战后更新评分，适合追踪可能随时间变化的实力，但结果会受比赛顺序和更新步长影响。对于固定版本的 LLM，评测期间通常可以假设其能力保持不变，因此可以用 BT 一次性拟合全部对战结果，估计一组固定的相对实力参数。这样的估计不依赖记录的排列顺序，并有便于构造置信区间的统计理论支持，更适合 Arena 的评分和排名估计。[参考：Sections 4–5](https://arxiv.org/html/2403.04132v1#S4)

#### Q6. [Limitations]

What are some limitations of Chatbot Arena? What are general limitations of human preference benchmarks? [Section: 8. Discussion]

**Answer:**

Chatbot Arena 的局限包括：参与者主要是 LLM 爱好者和研究人员，可能不能代表普通用户；问题来自在线聊天平台，未必反映真实生产环境或专业领域的需求；评测主要关注有用性（helpfulness），缺少对安全性（safety）的专门评估。[论文第 8 节](https://arxiv.org/html/2403.04132v1#S8)

人类偏好基准还受到主观判断、投票噪声和潜在恶意行为的影响。用户偏爱的回答未必更正确，例如流畅、自信的表达可能掩盖事实错误，因此偏好排名不能直接等同于模型的正确性或安全性。

### VibeCheck

#### Q7. [Problem]

What problem is VibeCheck aiming to solve? [Section: 1. Introduction]

**Answer:**

现有评测主要关注正确性或预先设定的评价维度，难以充分反映开放式任务中的用户偏好和具体场景需求。VibeCheck 旨在自动发现并量化不同 LLM 输出在语气、风格等方面的定性差异（vibes），帮助理解这些差异与用户偏好的关系，并辅助用户选择适合其任务的模型。[参考：Introduction](https://arxiv.org/html/2410.12851v1#S1)

#### Q8. [Current works]

What are existing approaches to evaluation in this space, and why are they not sufficient? [Section: 2. Related Work]

**Answer:**

- 基于预设维度的评测：事先规定正确性、清晰度、简洁性等评价维度，可能遗漏特定模型或任务中重要但未预先想到的特征。
- 两两比较与偏好预测：主要判断用户更喜欢哪个回答，但缺少可解释的分析来说明哪些具体特征与偏好相关。
- 使用 LLM 做定性分析：能够发现或描述输出差异，但部分工作缺少系统的量化验证，难以判断这些差异是否可靠、能否区分模型，以及是否与用户偏好相关。

[参考：Related Work](https://arxiv.org/html/2410.12851v1#S2)

#### Q9. [Proposed solution]

What are the inputs and outputs of VibeCheck? [Section: 3. Vibe-Based Evaluations, Briefly 5 for Examples]

**Answer:**

输入是一批提示词、模型 A 和 B 对每个相同提示词的回答，以及用于评估偏好相关性的用户偏好标签。部分实验在缺少人类偏好标签时使用 LLM 评委生成替代标签。

输出是一组自动发现的、可解释的差异维度（vibes），例如正式到友好的语气、幽默程度或回答结构，以及模型回答在这些维度上的比较分数和验证指标。这些指标衡量维度是否定义清晰、能区分模型，以及能否预测用户偏好。[参考：Sections 3–5](https://arxiv.org/html/2410.12851v1#S3)

#### Q10. [Method details]

How are vibes quantified? What is the metric of success (i.e., what numbers in the results section do we want to be high)? [3. Vibe-Based Evaluations]

**Answer:**

针对同一提示词的两个回答，评委在每个 vibe 上给出 $-1$、$0$ 或 $+1$：分别表示 A 比 B 更少体现该特征、两者相近或特征不适用、A 比 B 更体现该特征。分数表示特征方向，不直接表示回答优劣。

成功指标包括：

- Cohen’s Kappa：衡量不同评委的评分一致性，越高说明该维度的定义越清晰。
- Model-matching accuracy：利用 vibe 分数预测回答来自哪个模型，在留出数据上的准确率越高，说明这些维度越能区分模型。
- Preference prediction accuracy：利用 vibe 分数预测用户更喜欢哪个回答，在留出数据上的准确率越高，说明这些维度越能反映用户偏好。

单个 vibe 的可分性还可以用其比较分数的平均值衡量；绝对值越大，表示两个模型在该维度上的差异方向越一致。[参考：Section 3](https://arxiv.org/html/2410.12851v1#S3)

#### Q11. [Key insight / contributions]

Why does VibeCheck address the issues present in current approaches? How does VibeCheck address some limitations of Chatbot Arena? What are their contributions? (These could be methods, software artifacts, formalization of a problem, datasets, etc.) [Section 6. Application, 8. Conclusion]

**Answer:**

两个模型的传统准确性指标可能接近，但用户偏好仍可能不同。VibeCheck 自动发现并量化回答在风格、结构和表达方式上的差异，减少对预设评价维度的依赖，为正确性评测提供补充。Chatbot Arena 主要提供偏好结果和总体排名，VibeCheck 则进一步分析哪些输出特征与偏好相关，使模型比较更容易解释；这种相关性分析并不证明因果关系。

主要贡献包括：提出定义清晰、能区分模型、与用户偏好相关这三个可量化标准；构建自动发现、验证和迭代改进 vibes 的系统；通过与人类发现的差异对照、分析真实偏好数据，并在摘要、数学解题和图像描述等任务中验证方法的作用。[参考：Sections 3、5–6、8](https://arxiv.org/html/2410.12851v1#S6)

#### Q12. [Limitations]

What are some limitations of VibeCheck? Would there be circumstances where the results may be untrustworthy? [Section 7. Limitation]

**Answer:**

VibeCheck 难以区分相关性与因果性，例如用户偏爱详细的回答，实际原因可能是其内容更准确。验证成本也较高，因为每个评委都需要对每条样本的每个 vibe 评分。LLM 评委可能判断错误或偏爱自身输出；发现过程还可能生成重复维度、遗漏重要差异，多次运行的结果也可能不同，影响复现。

当任务超出评委的知识或推理能力，或存在未控制的混杂因素时，结果可能不可靠。例如，把准确性带来的偏好归因于回答风格会产生误导。两个模型准确性差距很大时，发现的 vibes 还可能主要集中在准确性相关特征上，未能覆盖其他重要差异。[参考：Section 7](https://arxiv.org/html/2410.12851v1#S7)

## Other Practical Resources

### Machine Learning Competitions: Kaggle

Kaggle hosts many ML competitions with a tier/medal system valued by industry. Despite being competitive, there’s a strong culture of code sharing that accelerates learning.

### Recommended Entry-Level Competition

- [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

### Other Relevant Competitions

- [Chatbot Arena (LMSYS)](https://www.kaggle.com/competitions/lmsys-chatbot-arena)
- [WSDM Cup Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)
- [LLM Detect AI-Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/leaderboard)
- [LLM Classification and Fine-Tuning](https://www.kaggle.com/competitions/llm-classification-finetuning/overview)
