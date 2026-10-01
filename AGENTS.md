# AGENTS.md — Why (we still need) linear regression

本文件是仓库 `why_linear_regression` 的 agent 指南。任何 agent 在本仓库工作时**必须先读本文件**。
它的目的：让 agent 能够**准确回答用户关于这本书的任何问题**——包括书的内容、结构、写作背景，
以及书中覆盖的 linear regression 知识性问题——并能在需要时安全地修改和重新编译本书。

---

## 1. 项目概览

一本教程书，中英双语两个独立 LaTeX 项目：

- **`why-linear-regression-zh/`** — 中文版（本体，用户主要工作版本）
- **`why-linear-regression/`** — 英文版（对照中文版的完整镜像，专业英文水准）
- 书名：**Why (we still need) linear regression -- a deep-dive for applied and data scientists**
- 作者：**Haoyi Xiong (熊昊一)**（北京）、**Lingkai Kong (孔令凯)**（西雅图）
- 定位：写给有线性代数与微积分基础的 applied/data scientists；所有结论从头推导；
  理论不追求教科书级严格（渐近正则条件一律从略）；书末有全部习题解答。
- 写作方式：正文由 Kimi K2.8 在 Kimi Work 桌面版 Agent 模式下写成；人类作者负责 outline、
  内容把关与「问什么、信不信」。

### 章节结构（两版一致，每章之间 `\clearpage`）

| 文件 | 内容 |
|---|---|
| `chapters/00_preface` | 前言：写给谁 / 缘起与七位学者影响 / AI 写作声明与免责声明 |
| `chapters/01_intro` | 引言：9 个要回答的问题、两百年的想法、记号约定 |
| `chapters/02_onedim` | 一维回归：斜率=相关系数、方差视角、方向性、因果性 |
| `chapters/03_highdim` | 高维闭式解 $\bhat=(\X^\top\X)^{-1}\X^\top\y$、投影、Gauss–Markov、为什么平方、ridge |
| `chapters/04_gd` | 梯度下降：不动点、精确解 $\bhat_t$、谱滤波、三档步长 regime |
| `chapters/05_gd2` | 欠定 $d>n$：最小范数解 $\widehat{\bbeta}'=\X^\top(\X\X^\top)^{-1}\y$、隐式正则化、伪逆五面孔 |
| `chapters/06_ggm` | 从估计到推断：协方差/精度矩阵、条件独立检验、稀疏性、lasso/enet/glasso/CLIME、debiased lasso |
| `chapters/07_fdr` | 规模化推断：BH 程序、FDR 控制、BH=自适应稀疏估计、knockoffs |
| `chapters/08_kernel` | 核方法：$\x_i=\phi(\z_i)$、核插值、push-through、随机特征、NTK、double descent |
| `chapters/09_epilogue` | 结语：八条线索收拢 + 全书路线图 TikZ 图 + 「最终答案」 |
| `chapters/10_solutions`（英文版）/ `09_solutions_zh`（中文版） | 全部习题的完整解答 |

### 编译

```bash
cd why-linear-regression-zh && latexmk -xelatex -interaction=nonstopmode main.tex
cd why-linear-regression  && latexmk -xelatex -interaction=nonstopmode main.tex
```

- 必须用 **xelatex**（文档类 `ctexart`，中文字体 PingFang SC，即使英文版也是）。
- 插图在 `figures/`，全部为 **PDF 矢量图**（`includegraphics` 引用），由 `make_figures.py` 生成；
  **不要修改原图**；如必须重生成，先备份。
- 参考文献是 `main.tex` 内嵌的 `thebibliography`（33 条），不是 `.bib`。
- 交付前检查：`grep -c '^!' main.log` 应为 0；Overfull 应只剩 <3pt 的肉眼不可见项。

---

## 2. 回答问题的准则

1. **语言跟随用户**；用户用中文问就用中文答（技术术语可夹英文），反之亦然。
2. **书内的内容优先引用书中章节**：「参见第 X 章 / Section Y」。引用前先用 Grep/Read
   在对应章节的 `.tex` 文件里核实，不要凭记忆引页码。
3. **区分三类知识**，并在答案中保持诚实：
   - (a) 书中明确覆盖的（可直接答，注明章节）；
   - (b) 书中没覆盖、但属标准教科书知识的（可以答，标注「书外补充」）；
   - (c) 需要查证的事实（最新论文、活着的学者近况等）——用联网搜索核实后再答。
4. **不夸大本书**：本书理论部分不追求严格（前言免责声明 (b)）；遇到做理论的读者，
   主动说明本书的假设从略。
5. 数学记号一律使用本书的宏（见 §3），公式用 LaTeX。
6. 修改任何 `.tex` 前先读原文件；改动后必须重新编译对应语言版本并确认 0 错误；
   两个版本内容需保持镜像一致（改中文版要同步改英文版，反之亦然）。

---

## 3. 本书记号（回答时统一使用）

```
n 样本量, d 特征维度, q 潜变量维度, X ∈ R^{n×d} 设计矩阵（行 = x_i^T）, y ∈ R^n
β ∈ R^d 参数, ε 误差;  b̂ / β̂ 最小二乘估计;  β̂' 欠定最小范数解;  β̂_t 第 t 步 GD 迭代
X^+ Moore–Penrose 伪逆;  κ = λ_max/λ_min 条件数;  η 学习率;  λ 正则参数
Σ 协方差矩阵, Θ = Σ^{-1} 精度矩阵;  r Pearson 相关系数, R² 判定系数
Var_n, Cov_n: 以 n 为分母的样本方差/协方差（无下标 = 总体量）
Φ = [φ(z_1)^T; ...; φ(z_n)^T] 特征矩阵;  K = ΦΦ^T 核 Gram 矩阵;  k(z) 核向量
s 稀疏度;  m 假设个数;  q (FDR 章) 目标假发现率;  p_(1)≤...≤p_(m) 排序 p 值
```

历史年份锚点：Legendre 1805 / Gauss 1809、1823 / Galton 1886 / Pearson 1896 /
Wright 1921 / Penrose 1955 / Reichenbach 1956 / Hoerl–Kennard 1970 (ridge) /
Kimeldorf–Wahba 1970 (representer) / Benjamini–Hochberg 1995 (FDR) /
Tibshirani 1996 (lasso) / Meinshausen–Bühlmann 2006 / Zou–Hastie 2005 (elastic net) /
Abramovich–Benjamini–Donoho–Johnstone 2006 (BH=自适应稀疏估计) /
Rahimi–Recht 2007 (随机特征) / Friedman–Hastie–Tibshirani 2008 (glasso) /
Cai–Liu–Luo 2011 (CLIME) / Zhang–Zhang 2014、van de Geer 2014 (debiased lasso) /
Barber–Candès 2015 (knockoffs) / Donoho 2006 (compressed sensing) /
Jacot–Gabriel–Hongler 2018 (NTK) / Belkin 2019 (double descent) / Lee 2019 (任意深度 NTK) /
Fan–Li–Zhang–Zou 2020 / Bach 2024 / Wu et al. 2025 (GD dominates ridge)。

---

## 4. 知识速查（每章核心结论，回答知识性问题的依据）

### 第 2 章 · 一维回归

- 无截距模型 $y_i = b x_i + \varepsilon_i$：$\hat b = \frac{\sum x_i y_i}{\sum x_i^2}$（凸性保证全局唯一）。
- 有截距：$\hat b = \frac{\Cov_n(x,y)}{\Var_n(x)}$，$\hat a = \bar y - \hat b \bar x$；回归线过数据中心点。
  $x_i\equiv 1$ 时 $\hat b=\bar y$ —— **最小二乘是均值的推广**。
- 标准化后 $\hat b = r\,\sigma_y/\sigma_x$（$r$ = Pearson 相关系数）。三件事：
  强度 $|r|$ = 陡度（标准差单位）；$r$ 对称、回归有方向；$R^2 = r^2$（拟合优度 = 被解释方差占比）。
- Galton 1886：$y = 0.516\,x + 33.73$ 英寸，$r\approx 0.5$ —— 「回归均值」是 $|r|<1$ 的算术，非生物学定律。
- 方差视角：$\Var(y) = b^2\Var(x) + \Var(\varepsilon)$（设 $\Cov(x,\varepsilon)=0$，**这已是方向性假设**）；
  相关性 = 方差的转移。
- 方向性：$y\sim x$ 与 $x\sim y$ 是两条不同的线；回归系数之积 $= r^2$，同一坐标系几何斜率之积 $=1$；
  椭圆点云上两条回归线都不是主轴（主轴 = PCA，最小化垂直距离）。
- 因果：$x\to y$、$y\to x$、共同原因 $z$ 三种结构产生**同一个 $r$；Reichenbach 共同原因原理给枚举、
  Wright 路径分析给「有外部因果知识时的演算」，数据本身永远不能判定方向。
  把系数读成因果效应需要识别策略（随机化/自然实验/无未测混杂）。

### 第 3 章 · 高维闭式解

- 正规方程 $\X^\top\X\,\bhat = \X^\top\y$；$\bhat=(\X^\top\X)^{-1}\X^\top\y$（$n\gg d$、满秩时唯一）。
- 几何：$\X\bhat$ 是 $\y$ 到列空间的正交投影；残差 $\perp$ 列空间。
- Gauss–Markov（Gauss 1823）：误差不相关、零均值、等方差下，OLS 是最优线性无偏（BLUE）。
- 为什么平方：凸二次 ⇒ 闭式唯一解；Gauss–Markov；与正态 MLE 的联系；Hessian $\X^\top\X\succeq 0$。
  绝对值损失无闭式（median regression），对异常值稳健但不可微、推断难。
- $n$ 靠近 $d$：方差先坏、秩后坏。修复 = ridge：
  $\bhat^{\mathrm{ridge}} = (\X^\top\X+\lambda\I)^{-1}\X^\top\y$，偏差 $-\lambda(\X^\top\X+\lambda\I)^{-1}\bbeta$
  （方向被拉向原点，幅度按 $1/(\text{特征值}+\lambda)$ 逐方向折扣）。SVD 视角：岭把奇异值 $s$ 收缩为
  $s^2/(s^2+\lambda)$ 因子。

### 第 4 章 · 梯度下降

- 迭代（平方损失）：$\widehat\bbeta_{t+1} = \widehat\bbeta_t - \eta\,\X^\top(\X\widehat\bbeta_t - \y)$；
  不动点条件 $\nabla L=0$ **就是正规方程**。
- 精确解（联立闭式解与迭代）：$\widehat\bbeta_t = \bhat + (\I-\eta\X^\top\X)^t(\widehat\bbeta_0-\bhat)$。
  一步到位求逆 = 用 $t$ 次多项式按揭出逆（Neumann 级数部分和）。
- 收敛：$0<\eta<2/\lambda_{\max}$；最优 $\eta^\star = 2/(\lambda_{\min}+\lambda_{\max})$，
  速率 $(\kappa-1)/(\kappa+1)$，步数 $O(\kappa\log(1/\varepsilon))$。「标准化特征」由经验升级为定理。
- 三档 $\eta$（图 5/图 6）：$\eta\le 1/\lambda_{\max}$ 单调下降；$1/\lambda_{\max}<\eta<2/\lambda_{\max}$
  震荡下降（每步变号、振幅衰减 $|1-\eta\lambda_{\max}|$）；$\eta>2/\lambda_{\max}$ 震荡发散。

### 第 5 章 · 欠定与隐式正则化

- $d>n$ 时解成流形 $\widehat\bbeta' + \mathcal N(\X)$：每点都是插值解与 GD 不动点；流形横向吸引、
  单点仅 Lyapunov 稳定；落点由初值的零空间分量决定。
- 最小范数解 $\widehat\bbeta' = \X^\top(\X\X^\top)^{-1}\y = \X^+\y$。
  **五个面孔是同一个解**：两个闭式公式、伪逆、GD 从零点极限、ridge $\lambda\to0^+$ 极限。
- 任意初值：$\lim_t\widehat\bbeta_t = \widehat\bbeta' + \P_{\mathcal N(\X)}\widehat\bbeta_0$（零空间中性）。
- $\X^\top\X$ 与 $\X\X^\top$ 非零特征值及其重数相同 ⇒ 「$d$ 个公式」与「$n$ 个公式」可互换（有效维数 $r$）。

### 第 6 章 · 从估计到推断

- $\bbeta = \Sigma_{XX}^{-1}\Sigma_{Xy} = -\Theta_{Xy}/\Theta_{yy}$，$\Var(\varepsilon)=1/\Theta_{yy}$。
- **精度矩阵零元 ⟺ 条件独立**：$\Theta_{ij}=0 \iff i\perp j\mid$ 其余；偏相关
  $\rho_{ij\cdot\cdot} = -\Theta_{ij}/\sqrt{\Theta_{ii}\Theta_{jj}}$；$\beta_j=0 \iff y\perp x_j\mid\x_{-j}$。
- 稀疏性四张脸：组合（NP-hard→稀疏可解）、几何（$\ell_0$ 球的凸包是 $\ell_1$ 球）、统计（有效自由度
  $\widehat{df}=\E\|\widehat\bbeta\|_0$）、哲学（稀疏是关于世界的假设）。
- 方法：lasso $\arg\min \frac1n\|\y-\X\bbeta\|^2+\lambda\|\bbeta\|_1$；elastic net
  $\lambda(\alpha\|\bbeta\|_1+(1-\alpha)\|\bbeta\|_2^2)$（相关变量组）；graphical lasso
  （$\ell_1$ 惩罚的似然直接估计全图）与 CLIME（$\bm S\Theta-\I\approx0$ 逐列线性规划，更良态、可并行、
  但统计效率略低）；nodewise lasso（Meinshausen–Bühlmann 估计图结构）。
- debiased lasso：$\widetilde\bbeta = \widehat\bbeta_{\mathrm{lasso}} + \frac1n\widehat\Theta\X^\top
  (\y-\X\widehat\bbeta_{\mathrm{lasso}})$，偏差项 $O_p(s\log d/n) = o_p(n^{-1/2})$ 时被噪声项支配，
  恢复逐系数渐近正态 ⇒ 可检验的 $p$ 值与置信区间。**estimate 说哪些边可能存在，inference 说哪些边显著。**

### 第 7 章 · FDR 控制

- 多重检验：FWER = $\Prob(V\ge1)$，FDR = $\E[V/\max(R,1)] \le$ FWER。Bonferroni $\alpha/m$：
  FWER 强控制但功效随 $m$ 崩塌。
- **BH 程序**：$\hat k = \max\{k : p_{(k)} \le qk/m\}$，拒绝 $p_{(1..k)}$；独立（或 PRDS）下
  $\mathrm{FDR}\le q$；Bonferroni 拒绝集 $\subseteq$ BH 拒绝集。一般相依用 BY 修正 $q/\sum_{j=1}^m j^{-1}$。
- BH = 自适应稀疏估计（Abramovich et al. 2006）：稀疏正态均值模型下，BH 阈值自动达到 oracle 阈值的
  minimax 速率 $2s\log(m/s)$（对 $2s\log m$）—— **控制 FDR 是稀疏性假设在检验问题上的对偶**。
- **Knockoffs**（Barber–Candès 2015）：在设计矩阵中造影子变量 $\tilde\X$；$W_j = |Z_j| - |\tilde Z_j|$，
  零假设下 $W_j \overset d= -W_j$（交换等变性）；knockoff+ 阈值
  $\min\{t : \frac{1+\#\{W_j\le -t\}}{\#\{W_j\ge t\}} \le q\}$。模型无关、有限样本精确 FDR 控制。
- 两路线对比：BH+debiased lasso 依赖渐近正态（近似保证）；knockoffs 靠交换对称（精确保证，
  构造需 $d\le n$ 或 group-expansion 技巧）。

### 第 8 章 · 核方法与深度网络

- $\x_i=\phi(\z_i)$；核 $k(\z,\z')=\langle\phi(\z),\phi(\z')\rangle$；$\K=\Phi\Phi^\top$（无 $\phi$ 坐标出现）。
  Mercer：正定核 ⟺ 某（可能无穷维）特征映射。
- $d>n$（核插值 / kernel ridgeless）：$\hat f(\tilde\z) = \k(\tilde\z)^\top\K^{-1}\y
  = \sum_i \alpha_i k(\tilde\z,\z_i)$，$\bm\alpha=\K^{-1}\y$。GD 从零点收敛点 = RKHS 最小范数插值函数
  （Kimeldorf–Wahba representer）。
- $n\gg d$：直接核化不可能（$\Phi^\top\Phi$ 依赖特征坐标系）；ridge + **push-through**
  $(\Phi^\top\Phi+\lambda\I_d)^{-1}\Phi^\top = \Phi^\top(\K+\lambda\I_n)^{-1}$ ⇒ 核岭回归
  $\hat f(\tilde\z)=\k(\tilde\z)^\top(\K+\lambda\I_n)^{-1}\y$；$\lambda\to0^+$ 两 regime 殊途同归。
  GP 回归 = 同一对象的贝叶斯叙述。
- 核的「天然性」：多项式核（有限维，省事）；高斯核（$d=\infty$，核化是唯一出路）；
  随机 Fourier 特征（Rahimi–Recht 2007，把核近似回有限维）。
- **桥一**：两层网络 $f(\z;\a)=\frac1{\sqrt D}\sum_j a_j\sigma(\omega_j^\top\z+b_j)$，冻结第一层、
  零初始化、GD 训练第二层 = Fourier 特征空间的 **ridgeless 回归**；RF 核
  $k_{\mathrm{RF}}\to\E[\sigma(\omega^\top\z+b)\sigma(\omega^\top\z'+b)]$。
- **桥二（NTK, Jacot et al. 2018）**：$\Theta_t(\z,\z')=\langle\nabla_\theta f(\z),\nabla_\theta f(\z')\rangle$；
  宽极限 $D\to\infty$ 下 $\Theta_t\to\Theta$ 全程不动 ⇒ 函数空间线性 ODE ⇒
  $f_\infty(\tilde\z)=\bm\Theta(\tilde\z)^\top\bm\Theta^{-1}\y$ —— 与核插值逐符号同构。
  任意深度：Lee et al. 2019。**参数空间的非凸是坐标系的幻象。**
- 务实注记：NTK = lazy regime（无特征学习，真实深度学习失效处理论在建）；double descent
  （Belkin 2019）：插值阈值后测试误差继续降，最小范数解是其最干净的线性化身；
  Wu et al. 2025：最优提前停止的 GD 在统计意义下支配最优 ridge。

### 第 9 章 · 结语（全书的「最终答案」）

线性回归是唯一同时做到四点的模型：**可精确求解**（凸、闭式、无超参数）、**可解释**
（每个系数、$R^2$、杠杆、自由度有几何意义）、**可审计**（每个定理有假设清单 + 具名诊断修复）、
**可推广**（换损失→稳健回归；加惩罚→岭/lasso；加特征→非线性；迭代加权→GLM；堆叠→神经网络）。
方法论口号：先拟合直线，审计假设，诚实披露弱点——然后才轮到更花哨的模型。

---

## 5. 常见问题回答口径（FAQ seeds）

- **「斜率为什么是相关系数？」**→ 第 2 章：$\hat b=\Cov_n/\Var_n = r\,\sigma_y/\sigma_x$；
  $r$ 是斜率的标准差单位形态；$R^2=r^2$。
- **「为什么不能从回归读出因果？」**→ 第 2 章：三种结构同一个 $r$；Reichenbach 枚举、Wright 演算、
  识别策略在数据之外。
- **「$d>n$ 训练误差为零凭什么泛化？」**→ 第 5 章（最小范数选择）+ 第 8 章（double descent、
  ridgeless、GD 隐式正则化支配 ridge）。
- **「核方法和线性回归什么关系？」**→ 第 8 章：对 $\x=\phi(\z)$ 线性 ⟺ 对 $\z$ 非线性；
  两个闭式解都是只用核组装的估计量；深度学习宽极限 = 线性回归的核化。
- **「lasso 和 ridge 怎么选？」**→ 第 3、6 章：偏差预算 vs 稀疏假设；相关变量组用 elastic net；
  估计图结构用 nodewise/glasso/CLIME；要 $p$ 值用 debiased lasso（第 6 章）或多重检验（第 7 章）。
- **「一页 $p$ 值该怎么 thresholds？」**→ 第 7 章：BH（FDR 控制、功效远优于 Bonferroni）；
  模型无关的精确控制用 knockoffs。
- **「这本书 cover 什么、不 cover 什么？」**→ 前言 (a)：GLM、稳健回归、混合效应、时间序列、
  贝叶斯路线、在线学习、联邦场景均未覆盖；只覆盖「不答就不诚实」的主线。

---

## 6. 编辑本书时的硬性约定

1. 改任一章内容 ⇒ **同步改另一语言版本**（结构镜像），并重新编译两个版本，均须 0 错误。
2. 章节标题给页眉短标题：`\section[短标题]{完整标题}`（英文标题全大写后极易超版心）。
3. 避免把 `\widehat{\bm{\beta}}`（即 `\bhat`）等含 `\bm` 的命令放进 `\section/\subsection` 的
   **可选参数**（hyperref 书签转换会报 `Illegal parameter number`）；可选参数里只允许 `$d>n$` 这类简单数学。
4. 宽表格用 `\resizebox{\textwidth}{!}{...}` 包裹（本书有多张三列对比表）。
5. 插图一律 PDF；正文不出现 `includegraphics` 引用 PNG/JPG。
6. 不要改动 `figures/` 下现有 PDF 与 `make_figures.py`，除非用户明确要求重生成插图。
7. 提交信息用英文简明描述；推送目标 `origin/main`（github.com/xhyccc/why_linear_regression）。
