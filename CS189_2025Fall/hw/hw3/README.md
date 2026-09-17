# HW3：反向传播与优化器

已完成书面题 Q1–Q10 和 Notebook Q1–Q4。答案用于个人学习；OpenAI Codex 协助讨论推导、撰写答案、实现代码和验证，书面文件与 Notebook 均记录了 AI 使用情况。

| 文件 | 内容 |
| --- | --- |
| [hw3_written.tex](hw3_written.tex) / [hw3_written.pdf](hw3_written.pdf) | 计算图与伴随量、优化器比较、AdamW 解耦、Adam 两步数值计算 |
| [hw3.ipynb](hw3.ipynb) | BearTensor 自动求导、SGD/Momentum/Adam、优化器对比和葡萄酒质量预测，含运行输出 |
| [validation/test_solutions.py](validation/test_solutions.py) | 从 Notebook 读取实际实现，运行内置用例和额外梯度检查 |

## 本地运行

建议使用 Python 3.12，在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-local.txt
.\.venv\Scripts\python -m jupyterlab
```

选择该环境并按顺序运行 Notebook。首次运行需要联网下载 OpenML 的 `wine-quality-red` 数据和 Typst 绘图包。`requirements.txt` 保留课程环境依赖，并补上绘图单元格使用的 `typst`；`requirements-local.txt` 提供运行本作业所需的较小环境。

```powershell
.\.venv\Scripts\python -m pytest validation -q
pdflatex -interaction=nonstopmode -halt-on-error hw3_written.tex
```

`predict(x)` 接收题目预处理后的 11 维标准化特征，返回 Python 标量。模型使用 16 个 sigmoid 隐藏单元和线性输出，以自行实现的 Adam 训练 200 轮，随机种子为 189；训练仅使用训练集标签。

矩阵乘法遵循模板中的列向量约定：一维输入先转为列向量，反向传播时恢复原形状；一维向量内积使用 `dot()`。Momentum 的 `velocities` 保存已乘学习率的更新位移，与书面题累积梯度的写法等价，符合内置测试约定。

## 验证记录

验证日期为 2026-09-17，使用 Python 3.12.7、NumPy 1.26.4、scikit-learn 1.5.1、Otter 6.1.3 和 Typst 0.15.0。

- Notebook 的全部计算单元格已顺序执行，Q1–Q3 的 25 个内置用例全部通过。Q4 模板没有公开测试用例，已直接验证预测值有限且全数据 MSE 不超过 2.0。
- `pytest validation -q` 的 19 项检查全部通过，包含上述内置用例、有限差分梯度、共享节点、重复反向传播、2,500 层计算图、矩阵乘法形状和极端 sigmoid 输入。
- 葡萄酒数据共 1,599 条，训练集 959 条、测试集 640 条。最终训练 MSE 为 0.3081，测试 MSE 为 0.3978，全数据 MSE 为 0.3440；本机训练耗时约 0.67 秒，不含下载和绘图。
- 书面 PDF 已编译并检查全部 7 页；Q8–Q10 放在同一页，便于核对连续更新。

按题目要求保留原始预处理代码，其中标准化器在全数据上拟合，因此上述测试 MSE 不代表完全隔离预处理的泛化评估。训练轮数固定，未按测试损失挑选模型。

末尾 `grader.export(...)` 是手动提交打包步骤，本次验证未执行；未向 Gradescope 提交。模板导入 Plotly 时，当前机器的 Kaleido 版本会发出兼容性提示，本作业图表使用 Matplotlib，已正常保存。生成的演示计算图、下载缓存和临时环境不提交到仓库。
