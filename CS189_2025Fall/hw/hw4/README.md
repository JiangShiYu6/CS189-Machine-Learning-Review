# HW4 答案与运行说明

已完成论文问答和 `hw4_part1.ipynb`、`hw4_part1_fixed.ipynb`、`hw4_part2.ipynb`。OpenAI Codex 协助答案整理、实现和实验验证，文件中记录了 AI 使用情况。

## 论文问答

[答案 PDF](hw4_paper_question_student.pdf) 与 [LaTeX 源文件](hw4_paper_question_student.tex) 包含全部 21 道题的英文答案，涵盖 ResNet、Transformer、BLEU/perplexity 和 Vision Transformer。按源文件题目顺序作答，移除了一个多余的无题目答案框；源文件与讨论截图的部分题号不同。

PDF 共 8 页，已编译并逐页检查公式、图示与分页。可在本目录运行 `pdflatex -interaction=nonstopmode -halt-on-error hw4_paper_question_student.tex` 重新生成。论文问答和编程文件已完成，Kaggle 与 Gradescope 的外部提交另行进行。

## 环境

使用 Python 3.12 和 NVIDIA GPU。在本目录建立环境后安装匹配的 CUDA 组件，再安装其他依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python -m pip install -r requirements-local.txt
.\.venv\Scripts\python -m jupyterlab
```

本机验证使用 RTX 4060 Laptop GPU（8GB 显存）。Part 2 保持有效 batch size 32，按每次 4 个样本累积梯度，并使用 CUDA 混合精度。训练轮数依次为 CNN 20 轮、ResNet-18 50 轮、Transformer 15 轮、DNABERT 5 轮、三种 ConvNeXt 各 5 轮。

## 数据与模板修正

- Mini-ImageNet 只下载覆盖原题前 10% 行的必要 parquet 分片，再按原题方式平衡抽样：训练 1,000 张、验证 200 张、测试 200 张。
- TinyStories 从完整训练集按种子 42 抽取 1,000 篇。先按故事划分训练/验证集，再生成窗口，避免同一故事跨越两个集合。
- 两个 Part 1 采用一致的实现。下一词预测使用 decoder-only 模式，避免完整编码器通过交叉注意力泄露未来词；位置编码注册为 buffer，词表排序固定，损失忽略 padding，文本生成正确处理起始 token 并调用传入的模型。
- DNABERT 固定为模板指定的修订版本，使用标准 BERT 加载相同权重，避免远程包装类的配置类型冲突。
- 两份音频 ZIP 在仓库中是 Git LFS 指针。解压单元格会从原始课程仓库下载实际归档到 `data/`，验证指针中的 SHA-256 后提取音频，不修改原指针文件。
- 音频采用单声道功率谱、相对峰值的 80 dB 转换和 ImageNet 通道归一化；训练、单样本展示和测试预测使用相同变换。音频文件按 CSV 的 `filename` 顺序预测，保持 ID 对齐。
- 音频播放器直接嵌入 Notebook，无需本机扬声器或 pygame。图表使用 Matplotlib，无需 Chrome/Kaleido。

## 验证与生成文件

```powershell
.\.venv\Scripts\python -m pytest validation -q
```

测试读取三个 Notebook 的实际实现，检查模型形状、梯度、因果掩码、位置编码、DNA token、音频变换和梯度累积。测试使用独立的 `validation/` 目录，避免遮挡 Otter 的内嵌测试。

Part 2 生成 `dna_test_predictions.csv`、`urbansound8k_test_predictions.csv`、`models/*.pt` 和 `classifier.joblib`。后者包含 DNA 模型与所选声音模型的 CPU state dict 以及元数据。预测前重新加载保存的 state dict；模型文件和下载缓存由 Git 忽略，但保留在本地。

Kaggle 上传、用户名、比赛分数截图及 Gradescope 提交需要实际提交记录。Notebook 的最后一个 `grader.export(...)` 单元格保留为手动打包步骤；本次计算验证不执行外部提交。

题目要求的声音划分是同一 fold 内按音频片段随机拆分，同一原始录音的片段可能同时出现在训练与验证集，因此其验证准确率不能当作严格按录音隔离的泛化指标。

## 实际运行结果

三个 Notebook 的编程题已补全并执行。实验按章节分段运行；两个 Part 1 采用相同实现，共享验证过的实验输出。没有将同一实验描述为两次独立训练。分类任务分别展示四张指标图，语言模型展示两张损失图；图表按日志中保留的小数精度重绘。

| 模型 | 轮数 | 最终训练准确率 | 最终验证准确率 |
| --- | ---: | ---: | ---: |
| CNN | 20 | 53.90% | 46.50% |
| ResNet18 | 50 | 90.80% | 70.00% |
| DNABERT | 5 | 67.13% | 43.00% |
| ConvNeXt scratch | 5 | 43.73% | 44.29% |
| ConvNeXt frozen | 5 | 86.74% | 83.57% |
| ConvNeXt full | 5 | 99.64% | 97.14% |

Transformer 训练 15 轮，最终训练/验证损失为 **3.5033 / 4.2997**，已生成三个提示词的续写。模型能生成常见故事句式，但存在重复和连贯性不足。DNABERT 的验证损失上升，说明已经过拟合；训练准确率达标不意味着泛化良好。

23 项功能测试通过，涵盖两个 Part 1 的实现、原始形状检查、反向传播、未来词隔离、位置编码、DNA 分词、真实 WAV 转换、分贝数值性质和梯度累积。所有计算单元格保存了运行结果，无错误输出；仅手动提交导出单元格未执行。

两份预测文件分别含 **1,377 条 DNA 预测**和 **175 条声音预测**，ID 与输入 CSV 一一对应，无缺失值。声音预测采用验证集表现最好的全量微调模型。模型权重和 `classifier.joblib` 保留在本地；未提交到 Kaggle 或 Gradescope。

逐轮数据见 [experiment_metrics.json](validation/experiment_metrics.json)，学习率实验记录见 [tuning_runs.json](validation/tuning_runs.json)。

## 声音分类后续修正

原先的 log1p 对很小的功率值近似线性，未充分展开弱频谱结构。现在对每张功率谱计算 `10 * log10(max(P, 1e-10))`，减去该图最大值，截断到 [-80, 0] dB，再映射到 [0, 1] 并进行 ImageNet 通道归一化。功率分贝系数和 80 dB 截断参考 [Torchaudio 官方文档](https://docs.pytorch.org/audio/2.3.0/generated/torchaudio.functional.amplitude_to_DB.html)。

三个 agent 分别审查了训练循环、官方资料和输入统计。每类前三个文件共 30 个样本的诊断中，各图“归一化后像素低于 0.01 的比例”的中位数由 94.75% 降到 13.88%；详见 [诊断数据](validation/spectrogram_diagnostics.json)。在同样的随机初始化、学习率 1e-4、关闭随机深度和 5 轮预算下，只改变输入表示，训练准确率由 12.72% 提高到 43.73%。这支持输入动态范围是先前训练停滞的主要原因。

三个声音模型均重建 Dataset 缓存并使用新变换重新训练，训练准确率均达到题目建议值。最终模型包记录 `audio_preprocessing=power_db_peak_relative_80db_imagenet_v1`，声音预测文件也由新模型重新生成。新增测试验证 20 dB 功率间隔、静音输入有限性和整体功率缩放不变性。
