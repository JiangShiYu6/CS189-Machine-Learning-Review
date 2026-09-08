# HW2: Chatbot Arena 与线性模型

本目录包含书面题、论文问答和已运行的 Notebook。答案用于个人学习；本次整理使用 OpenAI Codex 讨论推导、实现代码、调试和撰写说明，Notebook 中也记录了 AI 使用情况。

## 文件

| 文件 | 内容 |
| --- | --- |
| [written.tex](written.tex) / [written.pdf](written.pdf) | 残差、加权线性回归、Online K-means、MLE 与 MAP 的过程和答案 |
| [hw2_paper_answer.md](hw2_paper_answer.md) | Chatbot Arena 与 VibeCheck 的 Q1–Q12，附论文链接 |
| [arena_warmup.ipynb](arena_warmup.ipynb) | 数据筛选、两两胜率、分类排行榜、TF-IDF 聚类与提示词筛选实验 |
| [arena_style_control.ipynb](arena_style_control.ipynb) | Bradley–Terry 回归、Bootstrap、风格控制和自定义特征 |
| [legacy/arena_style_control.ipynb](legacy/arena_style_control.ipynb) | 旧版全部题目，包括额外的分类排行榜与分类风格控制 |
| [tests/test_solutions.py](tests/test_solutions.py) | 不依赖外部数据的函数行为测试 |

## 本地运行

建议使用 Python 3.12，在本目录打开终端：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-local.txt
.\.venv\Scripts\python -m jupyterlab
```

在 JupyterLab 中选择该环境，先运行 Warmup，再运行 Style Control；后者也包含独立的数据加载和筛选代码，不依赖另一个 Notebook 的内存状态。运行旧版时，以 `legacy` 为工作目录；它会从父目录导入绘图工具。

数据默认从公开的 [Arena Human Preference 100k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k) 下载，本次使用了 106,134 场对战。也可以设置 `HW2_ARENA_PARQUET` 为同一数据集的本地 Parquet 文件绝对路径，以避免重复下载。GPT-2 分词器首次运行需要联网下载词表。

Notebook 使用 Plotly 的交互输出，不依赖 Chrome 或 Kaleido；如果 GitHub 页面没有显示交互图，请在 JupyterLab 中查看。聚类 CSV 在运行时生成，下载的数据、虚拟环境和临时文件不提交到仓库。最后的 `grader.export(...)` 单元格仅用于需要打包作业时手动执行；本次完整计算验证未运行导出打包步骤。

编译书面题并运行额外测试：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error written.tex
.\.venv\Scripts\python -m pytest tests -q
```

## 实验记录

- Warmup 使用对战数最多的 20 个模型，聚类阶段从去掉最高频提示词后的数据中固定抽取 8,000 场。扫描 `K = 4, 6, 8, 10, 12`，根据归一化惯性曲线相对端点连线的偏离选择 `K = 8`。
- Q3e 的规则为“移除包含至少两个问号的提示词”，移除 581 场（7.2625%），每个模型仍至少有 385 场；GPT-4o-2024-05-13 从第 4 升至第 2。该规则经过探索选择，不能把这次排名变化解释为泛化能力或因果效应。
- 正式 Style Control 的新特征为“邀请继续提问”，例如 `let me know`、`feel free to ask`。25 次 Bootstrap 的平均系数约为 −0.224，区间约为 [−0.312, −0.135]，绝对值超过标题特征的 0.109。
- 旧版采用同一特征的反向定义“直接结束回答，不附通用追问邀请”，以满足旧题对正系数的要求。它的系数约为 0.238，高于标题特征的 0.116。旧版先去重再选模型，正式版先选模型再去重，故两版模型集合与结果不完全相同。
- Bootstrap 以原始对战为单位采样，始终把正向和反向特征记录放在一起，避免将同一票当作两个独立样本。主要习题保留 10 次重采样的默认值，特征探索使用 25 次；这些次数适合演示，但置信区间尾部估计较粗糙。

## 验证

验证日期：2026-09-08。运行环境为 Python 3.12.7、NumPy 1.26.4、pandas 2.2.2、scikit-learn 1.5.1、Plotly 5.24.1、datasets 3.6.0、tiktoken 0.11.0、Otter 6.1.3。

三个 Notebook 已按顺序执行计算单元格：Warmup 的 5 组、正式 Style Control 的 10 组、旧版的 15 组内置测试全部通过。另有 8 项测试通过，覆盖双向胜率、模型筛选、特征反对称性、风格归一化、区间排名和 Bootstrap 可重复性。自动测试不代替开放题人工评分；开放题中的数字来自本次运行。

`written.pdf` 已通过 LaTeX 编译与新增页面的排版检查。Online K-means 第 (b) 问保留原题，答案说明其全数据目标的单调性不成立，并给出 `0, 2, -2` 的反例。
