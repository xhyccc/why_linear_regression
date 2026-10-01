# Why (we still need) linear regression —— 给应用与数据科学家的第一性原理深挖

*English README: [README.md](README.md)*

一本第一性原理的中文教程：**为什么两百年过去了，直线仍是诚实数据分析的起点**。

作者：熊昊一 (Haoyi Xiong)、孔令凯 (Lingkai Kong)

## 这本书讲什么

这不是一本罗列线性回归公式的教科书，而是一次"从直觉走到推理（from estimate to inference）"的深度漫游。全书从一条一维直线出发，依次穿过：

| 章节 | 内容 |
| --- | --- |
| 前言 | 适合的读者、写作缘起、所受学者影响、AI 协作写作说明 |
| 第 1 章 | 立论：为什么直线仍是诚实的起点 |
| 第 2 章 | 一维回归：从回归视角看相关性，从方差视角看方向性与因果性 |
| 第 3 章 | 高维回归：$n \gg d$ 时最小二乘的闭式解 $\widehat\beta=(X^\top X)^{-1}X^\top y$ |
| 第 4 章 | 梯度下降视角：迭代、不动点，以及 $\widehat\beta_t$ 关于 $t$ 的精确解 |
| 第 5 章 | $d>n$ 时的另一个不动点：最小范数解 $\widehat\beta'=X^\top(XX^\top)^{-1}y$ 与伪逆 |
| 第 6 章 | 从估计到推理：协方差/精度矩阵、条件独立、lasso / elastic net / CLIME / graphical lasso、debiased lasso |
| 第 7 章 | 大规模推理：从稀疏估计到 FDR 控制（BH 与 knockoffs） |
| 第 8 章 | 非线性潜空间：线性回归作为天然核方法，桥接随机傅里叶特征与神经正切核（NTK） |
| 第 9 章 | 结语：回到"为什么还需要线性回归" |
| 习题解答 | 全部习题的完整解答 |

适合有一定数学基础（学过线性代数与微积分）、希望提升统计理解与建模判断力的 applied 与 data scientists。理论部分不追求测度论级别的严格——搞理论研究的读者不必读，没系统学过的读者可以通读一遍。

## 仓库结构

```
why-linear-regression-zh/   中文版（ctexart，正文中文）
why-linear-regression/      英文版（article + xeCJK，正文英文）
AGENTS.md                   本书知识问答指南（可据此向 AI 提问书中任何内容）
```

每个版本目录下：`main.tex`（主文件）、`chapters/`（各章源码）、`figures/`（全部插图为 PDF，由 `make_figures.py` 生成）。

## 快速开始

编译（需要 XeLaTeX 与标准 TeX Live 宏包）：

```bash
cd why-linear-regression-zh   # 或 why-linear-regression
latexmk -xelatex -interaction=nonstopmode main.tex
```

重新生成插图——书中的全部数值实验（梯度下降轨迹、double descent、FDR 曲线等）也由这个脚本复现：

```bash
cd figures
python make_figures.py
```

预编译好的 PDF：

- [中文版 main.pdf](why-linear-regression-zh/main.pdf)（54 页）
- [英文版 main.pdf](why-linear-regression/main.pdf)（58 页）

## 写作说明（诚实声明）

本书正文由 Moonshot AI 的 Kimi K2.8 模型在 Kimi Work 桌面版 Agent 模式下写成——它直接编写 LaTeX 源码、编译 PDF、渲染检查页面并逐章校对；人类专家（熊昊一）提供大纲、章节内容与自己多年的思考——尤其是他专门研究过的几个问题：欠定情形下梯度下降的精确解、潜空间投影下的核方法视角、精度矩阵与条件独立的联系。一句话分工：*人决定"问什么"和"信什么"，机器负责"写"和"查"。*

前言中致敬了七位学者：Francis Bach、Ryan Tibshirani、Jingfeng Wu、Arthur Jacot、Jianqing Fan、David Donoho、Emmanuel Candès——他们的工作分别影响了本书的核方法视角、估计到推理的路线、迭代即正则化、NTK、高维统计框架、压缩感知与 FDR 控制等内容。
