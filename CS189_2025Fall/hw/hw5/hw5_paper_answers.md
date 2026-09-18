# HW5 paper answers

Paper: Dong et al., *How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition*, ACL 2024. [Paper](https://arxiv.org/abs/2310.05492)

## Q1. Motivation

The paper studies how to improve mathematical reasoning, code generation, and general instruction following together through supervised fine-tuning (SFT). Unlike single-task learning, which can optimize one ability in isolation, this setting must balance tasks with different data requirements and avoid interference or forgetting when training on other tasks. The authors investigate how data amount, mixture ratio, model size, and training order affect this balance. [Paper, Section 1](https://arxiv.org/html/2310.05492v4#S1)

## Q2. Aligning models with human intent

Alignment here means teaching an LLM to understand and follow a user's request and provide an appropriate, useful response. SFT examples pair an instruction, possibly with additional input or conversation history, with a desired assistant response. Stanford Alpaca contains about 52,000 examples generated using text-davinci-003, stored as `instruction`, `input`, and `output`; a concrete example asks for three tips for staying healthy, with an empty input and an output listing health advice. ShareGPT, used to train Vicuna, contains user-shared conversations with ChatGPT, so an example is a sequence of human and assistant turns rather than only a single input-label pair. [Alpaca data and format](https://github.com/tatsu-lab/stanford_alpaca), [Vicuna and ShareGPT](https://www.lmsys.org/blog/2023-03-30-vicuna/)

## Q3. Main research findings

### RQ1. Scaling individual abilities

Mathematical reasoning generally improves as SFT data increases, while general instruction-following performance reaches a plateau after roughly 1,000 examples. Coding benefits from more data most clearly at 33B; the 7B and 13B coding curves are less consistently monotonic. Larger models generally perform better given the same amount of data, so neither one universal scaling curve nor the same optimal data budget describes all abilities. [Paper, Section 3.2](https://arxiv.org/html/2310.05492v4#S3.SS2)

### RQ2. Mixing abilities

With limited data, mixed-task SFT often improves performance relative to training each task alone, suggesting useful transfer. With abundant data, mixing can reduce individual-task performance, particularly general alignment. A plausible explanation is that additional tasks supply useful supervision when data is scarce but introduce competing training signals once an ability already has enough examples; this is an interpretation of the experiments, not proof of a particular gradient-level mechanism. [Paper, Section 3.3](https://arxiv.org/html/2310.05492v4#S3.SS3)

### RQ3. Data amount versus ratio

The absolute amounts of data matter more than the mixture ratio alone: keeping a ratio fixed does not keep performance fixed when the total amount changes. Ratio can matter when the sources overlap in the abilities they teach but differ in their distributions, as with coding examples in ShareGPT and CodeAlpaca. Thus the conclusion is not that ratios never matter; their effect depends on the contents and distributions of the datasets. [Paper, Section 3.4 and Section 4.2](https://arxiv.org/html/2310.05492v4#S3.SS4)

### RQ4. Training strategies

Multi-task training mixes all abilities throughout training, preserving specialized abilities reasonably well but creating conflicts with general alignment. Sequential training learns code, then math, then general alignment and can forget earlier abilities; mixed sequential training first mixes code and math, then trains general alignment, but still risks forgetting in its final stage. DMT adds a small amount of specialized replay data during the general stage and provides the best balance in the authors' comparisons, although it does not maximize every individual score. [Paper, Section 3.5](https://arxiv.org/html/2310.05492v4#S3.SS5)

## Q4. Experimental setup

The authors fine-tune LLaMA models with 7B, 13B, and 33B parameters. Mathematical SFT uses GSM8K augmented through rejection-sampling fine-tuning, coding uses CodeAlpaca instruction-code examples, and general alignment uses ShareGPT conversations; evaluation uses GSM8K test accuracy, HumanEval code-generation pass@1, and MT-Bench conversation scores, respectively. They vary training fractions (1, 1/4, 1/16, 1/64, and 1/256), compare single-source and mixed-source training, vary mixture ratios while controlling component amounts, and compare multi-task, sequential, mixed sequential, and DMT schedules. Their standard SFT setup uses three epochs, batch size 16, and a peak learning rate of 2e-5. [Paper, Section 3.1 and Appendix C](https://arxiv.org/html/2310.05492v4#S3.SS1)

## Q5. Dual-stage mixed fine-tuning

DMT first trains on the full mixture of specialized math and coding data. It then trains on general instruction-following data mixed with a small retained sample of the specialized data. The first stage accommodates the larger data requirements of specialized abilities; the second develops general alignment while replaying old skills to reduce forgetting. Keeping specialized replay small also reduces the interference that can arise when all full datasets are mixed together. [Paper, Section 3.5](https://arxiv.org/html/2310.05492v4#S3.SS5)

## Q6(a). Advantages and disadvantages of generated data

LLM-generated examples can be produced cheaply at scale, cover diverse instructions, and adapt to real requests found in user logs. However, their answers may contain mistakes, bias, or repetitive styles, and training on them can propagate the teacher's weaknesses rather than add reliable knowledge. User logs also require permission and privacy filtering. Quality checks, deduplication, and diversity controls are therefore necessary; a large synthetic dataset is not automatically a high-quality dataset.

## Q6(b). Generated replay to reduce forgetting

Before specializing, freeze a copy of the original model and have it generate varied general instructions and corresponding responses, creating a replay set without accessing another external dataset. Mix a small, filtered portion of that set into math and coding SFT to preserve the original model's behavior on those instructions. When training tasks sequentially, also retain real examples from the earlier specialized dataset, or generate and verify additional examples of that task, then replay them while learning the next task. Generated replay can reduce forgetting, but cannot guarantee preservation of skills absent from its prompts or correct errors already made by the teacher.

GenAI acknowledgement: OpenAI Codex assisted in drafting and checking these answers against the paper and dataset documentation. Q6 is a proposed approach, not an experiment reported as having been run in the paper.
