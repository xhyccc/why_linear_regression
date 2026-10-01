# Why (we still need) linear regression —— 给应用与数据科学家的第一性原理深挖

中文版《Why (we still need) linear regression — a deep-dive for applied and data scientists》。

作者：熊昊一 (Haoyi Xiong)、孔令凯 (Lingkai Kong)

**立论：** 两百年过去了，直线仍是诚实数据分析的起点。本书从一维直线出发，依次经过高维闭式解、梯度下降的精确解与不动点（$n\gg d$ 与 $d>n$ 两个方向）、协方差/精度矩阵与条件独立（lasso、elastic net、CLIME、debiased lasso）、FDR 控制（BH、knockoffs），最后走到非线性潜空间下的核方法视角——以随机傅里叶特征与神经正切核（NTK）桥接深度网络。全部习题附完整解答。

适合学过线性代数与微积分、希望提升统计理解与建模判断力的 applied 与 data scientists。理论部分不追求测度论级别的严格：搞理论的读者不必读，没系统学过的读者可以通读一遍。

## 目录结构

- `main.tex` —— 主文件（ctexart）
- `chapters/00_preface_zh.tex` … `chapters/09_epilogue_zh.tex` —— 九章正文
- `chapters/09_solutions_zh.tex` —— 全部习题解答
- `figures/` —— 全部插图为 PDF，由 `make_figures.py` 生成
- `main.pdf` —— 预编译版，54 页
- 英文版见 `../why-linear-regression/`
- 全书知识问答指南见 `../AGENTS.md`

## 编译

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```

重新生成插图：

```bash
cd figures && python make_figures.py
```

## 写作说明

本书正文由 Moonshot AI 的 Kimi K2.8 模型在 Kimi Work 桌面版 Agent 模式下写成——直接编写 LaTeX、编译 PDF、渲染检查页面并逐章校对；人类专家（熊昊一）提供大纲、章节内容与自己多年的思考，尤其是欠定情形梯度下降精确解、潜空间投影下的核方法视角、精度矩阵与条件独立等他专门研究过的点。前言致敬了 Francis Bach、Ryan Tibshirani、Jingfeng Wu、Arthur Jacot、Jianqing Fan、David Donoho、Emmanuel Candès 七位学者。详见前言。
