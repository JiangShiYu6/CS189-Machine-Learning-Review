# HW5：SFT 数据组成与模型微调

本地论文问答、微调实验、评估、PDF 和代码提交包已完成。实验使用作业指定的 `Qwen/Qwen2.5-0.5B-Instruct`，比较原模型、只用机器学习题训练的方案和混合数据训练的方案。Kaggle 已完成逾期提交并评分，成绩截图原文件仍待嵌入报告，详见下方说明。

## 文件

第二轮改进的预测文件为 [submission_v2.csv](submission_v2.csv)，实验比较和复现步骤见 [改进记录](improvements_v2.md)。原来的 `submission.csv`、Notebook 和 PDF 保留第一轮结果，新的脚本与逐题记录位于本目录及 `results_v2/`。重新提交 Kaggle 时请选择带 `_v2` 的文件；新版截图显示 Score **0.42352**，Public score **0.40476**。

| 文件 | 内容 |
| --- | --- |
| [论文答案](hw5_paper_answers.pdf) · [可编辑原文](hw5_paper_answers.md) | Q1–Q6，含四个 RQ 和 Q6 的两个子题 |
| [Notebook](finetuning_tutorial.ipynb) · [PDF](finetuning_tutorial.pdf) | 完整代码、实际输出和损失曲线 |
| [实验报告](hw5_writeup.pdf) | 方法、对照实验、真实遗忘与能力提升案例 |
| [预测文件](submission.csv) | 隐藏测试集的 169 条模型预测 |
| [代码提交包](hw5_submission.zip) | Notebook、PDF、辅助代码和复现文件 |
| [实验记录](results/) | 逐题预测、汇总指标、训练日志、数据来源和案例 |
| [验证脚本](validate_hw5.py) | 检查数据隔离、预测、案例和代码一致性 |

## 复现

在独立 Python 环境中安装 CUDA 版 PyTorch 和本目录的依赖，然后在 GPU 上从头运行 Notebook。Colab 中上传 Notebook、`requirements.txt`、`hw5_sample_eval.csv` 和 `kaggle_test.csv` 后选择 GPU；Notebook 会下载固定版本的模型和数据。这里将 Transformers 从原要求的 4.57.2 更新为 4.57.3，修复前一版本读取本地 tokenizer 时的错误。

```bash
pip install -r requirements.txt
```

Notebook 内嵌了全部训练函数，与 `sft_pipeline.py` 一致。训练完成后可重新导出并检查：

```bash
python export_deliverables.py
python validate_hw5.py
python package_submission.py
```

模型权重和下载缓存不提交到 Git；从头运行可重新生成 `models/narrow`、`models/final`。精确数据与模型版本记录在 `source_revisions.json`，运行环境记录在 `results/manifest.json`。不同 GPU 和依赖版本可能带来浮点差异。

## 实际结果

| 模型 | 公开 CS189 题 | 专业留出题 | 通用留出题 |
| --- | --- | --- | --- |
| 原模型 | 6/25（24.0%） | 30/75（40.0%） | 73/142（51.4%） |
| 仅机器学习题 | 12/25（48.0%） | 26/75（34.7%） | 71/142（50.0%） |
| 最终混合方案 | 10/25（40.0%） | 28/75（37.3%） | 72/142（50.7%） |

混合方案改善了公开课程题表现，通用准确率下降一题，专业留出准确率仍低于原模型。因此不能声称所有能力都得到提升。小数据方案虽然公开准确率更高，但在两个独立留出组上都更差，报告保留了这次尝试及其局限。

两次训练在 RTX 4060 Laptop 8GB 上分别耗时约 157 秒和 659 秒。报告包含实际损失曲线、两例原先正确但微调后错误的输入，以及两例微调后答对的非 CS189 输入。`submission.csv` 的 169 个 ID、列顺序和 A–E 标签已检查。

重载模型时发现 Trainer 自动应用的混合精度上下文会影响 LoRA 计算，因此推理函数显式使用相同的 bfloat16 autocast。修复后重新核对了三个模型的评估预测和隐藏预测，共 895 条，字母结果全部复现；记录见 `results/reload_check.json`。保留本地权重时可运行 `python validate_hw5.py --reload-model` 检查重载。

## 数据与评估约定

沿用 starter 的约定，把 MMLU 的 `test` split 用作训练候选，另外保留 `dev` 与 `validation` 用于评估，因此本目录的结果不是标准 MMLU benchmark 分数。训练集共有 907 题，来自机器学习、计算机科学、大学数学、统计、世界历史、地理、营养学和初等数学；独立留出题有 217 道。

公开课程评估题有 25 道，其中 21 道也出现在隐藏测试文件中。训练候选中匹配课程题目的 7 道重叠题已排除，并进一步检查训练题与留出题不重叠。公开准确率只作描述，不用于选择参数；隐藏测试答案不可见。

所有模型统一使用相同的选择题 prompt 和受约束字母解码：补上回答前缀 `\boxed{`，比较有效选项字母的模型 logits，再格式化为 boxed answer。案例中的输出是这种解码得到的预测，不是模型自由生成的推理文字。小数据方案与混合方案同时改变了数据覆盖和训练预算，因此不能把差异完全归因于混合数据。

## Kaggle

本次采用用户提供的 `submission_v2.csv` 成绩截图：账号 `Shiyu63`，**Score 0.42352**，**Public score 0.40476**。截图没有将 Score 明确标为 Private Score，也未显示截止日期状态，因此按原字段记录。详见 `results_v2/kaggle_submission.json`。

上一版 `submission.csv` 的 Public Score 为 0.40476、Private Score 为 0.35294，历史记录保留在 `results/kaggle_submission.json`。

最新成绩已写入报告。对话附件目前没有可读取的本地图片路径，PDF 尚未嵌入原图；将这张新版截图保存为 `results_v2/kaggle_score.png` 后，运行 `python export_deliverables.py` 和 `python package_submission.py` 即可补齐。Gradescope 提交由本人完成。

OpenAI Codex 协助完成代码、调试、论文问答与报告整理；实验指标和逐题输出来自实际运行。
