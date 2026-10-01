#!/usr/bin/env python3
"""为《Why (we still need) linear regression》生成全部插图。
零依赖：任何标准 matplotlib/numpy/scipy 环境均可复现书中全部插图与数值实验。
优先使用 managed runtime 的 setup_plot（含 CJK 字体），缺失时自动退回通用 rcParams。
固定随机种子，输出 figures/。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # 优先使用托管运行时样式（含 CJK 字体）
    sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:  # 零依赖回退：图照常生成，使用通用 rcParams + 系统中文字体
    import matplotlib.font_manager as fm
    cjk = [f for f in ("PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
                       "Noto Sans CJK SC", "SimHei", "WenQuanYi Micro Hei")
           if f in {x.name for x in fm.fontManager.ttflist}]
    plt.rcParams.update({
        "font.size": 11,
        "font.sans-serif": cjk + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    })

OUT = Path(__file__).parent
rng = np.random.default_rng(20261001)

# 统一配色（沉稳学术风）
C_DATA = "#4C72B0"
C_FIT = "#C44E52"
C_AUX = "#55A868"
C_GRAY = "#7F7F7F"


def save(fig, name):
    name = name.replace(".png", ".pdf")   # 矢量输出
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- ch1 Galton
def ch1_galton():
    n = 300
    x = rng.normal(172, 6, n)                     # 父亲身高 cm
    r = 0.5
    y = 172 + r * (x - 172) + rng.normal(0, 5.2, n)  # 儿子身高，向均值回归
    b = r * y.std() / x.std()
    a = y.mean() - b * x.mean()
    xs = np.linspace(150, 196, 10)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(x, y, s=14, alpha=0.35, color=C_DATA, edgecolors="none",
               label="父子身高数据（模拟）")
    ax.plot(xs, xs, "--", color=C_GRAY, lw=1.4,
            label=r"$y=x$（完全遗传）")
    ax.plot(xs, a + b * xs, "-", color=C_FIT, lw=2.2,
            label=rf"回归线 $\hat{{y}}={a:.1f}+{b:.2f}\,x$")
    ax.annotate("斜率 $=r\\,\\sigma_y/\\sigma_x<1$：\n高的父亲，儿子\n平均没那么高",
                xy=(184, a + b * 184), xytext=(188, 158),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_FIT, lw=1.2),
                color=C_FIT)
    ax.set_xlabel("父亲身高 $x$ (cm)")
    ax.set_ylabel("儿子身高 $y$ (cm)")
    ax.set_title("Galton 的向均值回归：回归线比 $y=x$ 平")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(alpha=0.25)
    save(fig, "ch1_galton.png")


# ---------------------------------------------------------------- ch2 三联图
def ch2_leastsq():
    n = 40
    x = rng.uniform(0, 10, n)
    b_true = 1.6
    y = 3 + b_true * x + rng.normal(0, 1.6, n)
    b_hat = np.polyfit(x, y, 1)
    b_grid = np.linspace(0.2, 3.0, 200)
    rss = [np.sum((y - (np.polyval(b_hat, x) - b_hat[0] * x + bb * x)) ** 2)
           for bb in b_grid]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    ax.scatter(x, y, s=22, color=C_DATA, edgecolors="none", alpha=0.7)
    xs = np.linspace(0, 10, 10)
    ax.plot(xs, np.polyval(b_hat, xs), color=C_FIT, lw=2.2, label=r"最小二乘直线")
    for xi, yi in list(zip(x, y))[::6]:
        ax.plot([xi, xi], [yi, np.polyval(b_hat, xi)], color=C_GRAY, lw=0.9,
                alpha=0.8)
    ax.set_title("(a) 拟合：残差平方和最小")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(b_grid, rss, color=C_FIT, lw=2.2)
    ax.axvline(b_hat[0], color=C_GRAY, ls="--", lw=1.2)
    ax.plot([b_hat[0]], [np.sum((y - np.polyval(b_hat, x)) ** 2)], "o",
            color=C_FIT, ms=7)
    ax.annotate(r"$\hat{b}=\mathrm{Cov}(x,y)/\mathrm{Var}(x)$"
                f"\n$={b_hat[0]:.2f}$",
                xy=(b_hat[0], np.sum((y - np.polyval(b_hat, x)) ** 2)),
                xytext=(0.6, rss[0] * 0.55), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_GRAY))
    ax.set_title("(b) 目标函数：单峰、可解")
    ax.set_xlabel("斜率 $b$"); ax.set_ylabel("残差平方和")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.annotate("", xy=(7.2, 8.2), xytext=(0.6, 0.8),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))
    ax.text(7.0, 9.0, r"$\mathbf{y}\in\mathbb{R}^n$", fontsize=13)
    ax.annotate("", xy=(6.4, 1.7), xytext=(0.6, 0.8),
                arrowprops=dict(arrowstyle="-|>", color=C_FIT, lw=1.6))
    ax.text(5.6, 0.3, r"$\mathbf{x}\,b$（一维子空间）", fontsize=12, color=C_FIT)
    ax.annotate("", xy=(6.4, 1.7), xytext=(7.2, 8.2),
                arrowprops=dict(arrowstyle="-|>", color=C_AUX, lw=1.6))
    ax.text(7.8, 4.6, r"残差 $\perp$ 子空间", fontsize=11, color=C_AUX)
    ax.plot([6.4], [1.7], "o", color="k", ms=6)
    ax.text(6.5, 2.6, r"$\mathbf{x}\hat{b}$（投影）", fontsize=11)
    ax.set_title("(c) 几何：最小二乘 $=$ 正交投影")
    fig.tight_layout()
    save(fig, "ch2_leastsq.png")


# ---------------------------------------------------------------- ch4 GD
def ch4_gd():
    # 二次目标 L(b)=1/2 b^T H b，H 特征值 1 与 0.04（条件数 25）
    th = np.linspace(0, 2 * np.pi, 300)
    A = np.array([[1.0, 0], [0, 0.2]])
    R = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    H = R @ (A ** 2) @ R.T

    def run_gd(eta, steps, b0):
        b = np.array(b0, dtype=float)
        traj = [b.copy()]
        for _ in range(steps):
            b = b - eta * (H @ b)
            traj.append(b.copy())
        return np.array(traj)

    def contours(ax):
        w, v = np.linalg.eigh(H)
        for c in [0.2, 0.8, 1.8, 3.2, 5.0]:
            e = v @ np.diag(np.sqrt(c / w)) @ np.vstack([np.cos(th), np.sin(th)])
            ax.plot(e[0], e[1], color=C_GRAY, lw=0.8, alpha=0.55)
        ax.plot(0, 0, "k*", ms=11, label="最小二乘解 $\\widehat{\\beta}$")

    b0 = (3.5, 2.6)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7))

    def style(ax, title, legend_loc="upper left"):
        ax.set_xlim(-4.6, 4.6); ax.set_ylim(-3.4, 4.0)
        ax.set_title(title)
        ax.set_xlabel("$\\beta_1$")
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc=legend_loc)

    # (a) 单调下降：eta=0.9，轨道不跨谷底中线
    ax = axes[0]
    contours(ax)
    tr = run_gd(0.9, 60, b0)
    ax.plot(tr[:, 0], tr[:, 1], "-o", color=C_AUX, ms=3.5, lw=1.5,
            markevery=6, label="$\\eta=0.9$：不变号逼近")
    ax.plot([b0[0]], [b0[1]], "o", color="k", ms=6, mfc="none",
            label="初始化 $\\widehat{\\beta}_0$")
    ax.annotate("轨道始终位于谷底同一侧：\n误差投影不变号，慢但稳",
                xy=(-0.5, 0.8), xytext=(-4.35, 3.25), fontsize=10,
                color=C_AUX, arrowprops=dict(arrowstyle="->", color=C_AUX))
    style(ax, "(a) 单调：$0<\\eta\\leq 1/\\lambda_{\\max}$")
    ax.set_ylabel("$\\beta_2$")

    # (b) 震荡下降：eta=1.5，轨道每步跨谷底翻号（之字轨）
    ax = axes[1]
    contours(ax)
    tr = run_gd(1.5, 30, b0)
    ax.plot(tr[:, 0], tr[:, 1], "-o", color=C_FIT, ms=4, lw=1.7,
            markevery=1, label="$\\eta=1.5$：之字形收敛")
    for k, off in [(0, (8, 2)), (1, (-14, 8)), (2, (8, -2)), (3, (-16, -12))]:
        ax.annotate(f"$t={k}$", xy=(tr[k, 0], tr[k, 1]),
                    xytext=off, textcoords="offset points",
                    fontsize=8.5, color=C_FIT)
    ax.annotate("每步跨过谷底中线翻号，\n幅度以 $|1-\\eta\\lambda_{\\max}|=0.5$ 衰减",
                xy=(tr[2, 0], tr[2, 1]), xytext=(-4.35, -2.9), fontsize=10,
                color=C_FIT, arrowprops=dict(arrowstyle="->", color=C_FIT))
    style(ax, "(b) 震荡下降：$1/\\lambda_{\\max}<\\eta<2/\\lambda_{\\max}$")

    # (c) 震荡不收敛：eta=2.6，翻号放大
    ax = axes[2]
    contours(ax)
    b0d = (0.9, 0.7)          # 小初值：前数步留在视野内
    tr = run_gd(2.6, 7, b0d)   # ηλmax>2，每步翻号、幅度 ×|1-2.6|=1.6
    ax.plot(tr[:, 0], tr[:, 1], "-o", color="#8172B3", ms=4.5, lw=1.8,
            label="$\\eta=2.6$：发散之字")
    for k in range(len(tr)):
        ax.annotate(f"$t={k}$", xy=(tr[k, 0], tr[k, 1]),
                    xytext=(6, 5), textcoords="offset points", fontsize=8.5,
                    color="#8172B3")
    ax.annotate("每步翻号、幅度放大 $|1-\\eta\\lambda_{\\max}|=1.6$ 倍\n"
                "数步之内飞出视野",
                xy=(tr[-1, 0], tr[-1, 1]), xytext=(-4.35, -2.9), fontsize=10,
                color="#8172B3",
                arrowprops=dict(arrowstyle="->", color="#8172B3"))
    style(ax, "(c) 震荡不收敛：$\\eta\\geq 2/\\lambda_{\\max}$")

    fig.tight_layout()
    save(fig, "ch4_gd.png")


# ---------------------------------------------------------------- ch4 带符号误差与损失
def ch4_loss():
    """三档 eta：带符号误差（左轴，线性）与损失（右轴，对数）。

    关键事实：损失 = 各方向误差的偶次幂加权和，永不震荡；
    震荡只出现在迭代轨道的符号里（带符号误差）。
    """
    A = np.array([[1.0, 0], [0, 0.2]])
    R = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    H = R @ (A ** 2) @ R.T
    _, V = np.linalg.eigh(H)
    vmax = V[:, -1]              # 最大特征值方向
    b0 = np.array([3.5, 2.6])

    def series(eta, T):
        b = b0.copy()
        ss, ls = [vmax @ b], [0.5 * b @ H @ b]
        for _ in range(T):
            b = b - eta * (H @ b)
            ss.append(vmax @ b)
            ls.append(0.5 * b @ H @ b)
        return np.array(ss), np.array(ls)

    cases = [
        (0.5, 40, "单调下降",
         "$\\eta\\,\\lambda_{\\max}=0.5\\leq 1$：\n误差不变号，损失单调下降",
         C_AUX, None),
        (1.5, 40, "震荡下降",
         "$1<\\eta\\,\\lambda_{\\max}=1.5<2$：\n带符号误差每步翻号成锯齿；\n"
         "但损失（灰虚线）仍单调——\n偶次幂消灭了符号",
         C_FIT, None),
        (2.6, 12, "震荡不收敛",
         "$\\eta\\,\\lambda_{\\max}=2.6>2$：\n每步翻号、幅度\n放大 $|1-\\eta\\lambda_{\\max}|=1.6$ 倍",
         "#8172B3", (-1500, 1500)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    for k, (ax, (eta, T, name, lab, c, ylim)) in enumerate(zip(axes, cases)):
        ss, ls = series(eta, T)
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        l1, = ax.plot(ss, "-o", color=c, ms=3.5, lw=1.8,
                      markevery=max(1, len(ss) // 16),
                      label="带符号误差 $v_{\\max}^{\\top}(\\widehat{\\boldsymbol{\\beta}}_t-\\widehat{\\boldsymbol{\\beta}})$")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"({'abc'[k]}) {name}：$\\eta={eta}$", fontsize=12)
        ax.set_xlabel("迭代步数 $t$")
        ax.grid(alpha=0.3)
        axr = ax.twinx()
        l2, = axr.semilogy(ls, "--", color="0.45", lw=1.3,
                           label="损失（对数刻度）")
        axr.tick_params(axis="y", labelsize=8, colors="0.35")
        if k == 2:
            axr.set_ylabel("损失 $L(\\widehat{\\boldsymbol{\\beta}}_t)-L_{\\min}$（对数）",
                           fontsize=9, color="0.35")
        ax.legend(handles=[l1, l2], fontsize=8, loc="upper right",
                  framealpha=0.9)
        ax.text(0.04, 0.04, lab, transform=ax.transAxes, fontsize=9,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=c, alpha=0.92))
    axes[0].set_ylabel("带符号误差（线性刻度）")
    fig.tight_layout()
    save(fig, "ch4_loss.png")


# ---------------------------------------------------------------- ch5 隐式正则化
def ch5_implicit():
    n, d = 30, 100
    X = rng.normal(size=(n, d))
    beta_true = np.zeros(d)
    beta_true[:5] = [3, -2, 1.5, 1, -0.8]
    y = X @ beta_true

    # 谱衰减 + GD 极限曲线
    sv = np.linalg.svd(X, compute_uv=False)
    lam = sv ** 2
    kappa = np.linspace(0, 1.5, 200)  # （保留：备用网格）
    # 解析：beta_hat(kappa) = V diag(1-(1-kappa*lam)^T) V^T X^T y, 取 T=10000 近似不动点
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    XTy = X.T @ y
    T = 4000
    eta = 1.2 / lam[0]
    lim = np.array([1 - (1 - eta * l) ** T for l in lam])
    beta_gd = Vt.T @ ((U.T @ y) * lim / s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    idx = np.argsort(np.abs(beta_gd))[::-1][:12]
    ax.bar(np.arange(12) - 0.2, beta_true[idx], width=0.4, color=C_GRAY,
           label="真实 $\\beta^*$（5 个非零）")
    ax.bar(np.arange(12) + 0.2, beta_gd[idx], width=0.4, color=C_FIT,
           label="GD 极限 $\\widehat{\\beta}_{\\mathrm{GD}}$（零初始化）")
    ax.set_xticks(np.arange(12), [f"$\\beta_{{{i+1}}}$" for i in range(12)],
                  fontsize=9)
    ax.set_title("(a) GD 自动找到「能量集中」的插值解")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    # 谱滤波视角：g(λ)=1-(1-ηλ)^T，不同迭代步数 T
    lv = np.linspace(0, 1, 300)  # λ/λ_max
    for T, c in [(50, C_GRAY), (400, C_AUX), (4000, C_FIT)]:
        g = 1 - (1 - lv) ** T
        ax.plot(lv, g, lw=2.0, color=c, label=rf"迭代 $T={T}$")
    ax.text(0.35, 0.55, r"$g(\lambda)=1-(1-\eta\lambda)^T$", fontsize=11)
    ax.set_xlabel("相对特征值 $\\lambda/\\lambda_{\\max}$")
    ax.set_ylabel("谱滤波 $g(\\lambda)$")
    ax.set_ylim(-0.03, 1.15)
    ax.set_title("(b) GD 极限 $=$ 谱域「低通滤波器」")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "ch5_implicit.png")


# ---------------------------------------------------------------- ch6 lasso 路径
def ch6_lassopath():
    n, d = 60, 8
    X = rng.normal(size=(n, d)) / np.sqrt(n)   # 单位尺度列，X_j^T X_j / n ≈ 1
    beta = np.array([4, -3, 2.5, 0, 0, 1.5, 0, 0.0])
    y = X @ beta + rng.normal(0, 0.3, n)

    # 坐标下降求 lasso 路径：min (1/2n)||y-Xb||^2 + λ||b||_1
    # 更新：b_j = S(X_j^T r / n, λ) / (X_j^T X_j / n)
    z = np.sum(X ** 2, axis=0) / n
    lambdas = np.geomspace(1.5, 0.004, 60)
    B = np.zeros((len(lambdas), d))
    b = np.zeros(d)
    for k, lam in enumerate(lambdas):
        for _ in range(400):
            for j in range(d):
                r = y - X @ b + X[:, j] * b[j]
                rho = X[:, j] @ r / n
                b[j] = np.sign(rho) * max(abs(rho) - lam, 0) / z[j]
        B[k] = b

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    nonzero = [j for j in range(d) if beta[j] != 0]
    zero = [j for j in range(d) if beta[j] == 0]
    for j in nonzero:
        ax.plot(lambdas, B[:, j], lw=2.0,
                label=rf"$\beta_{{{j+1}}}^*={beta[j]}$（真实非零）")
    for j in zero:
        ax.plot(lambdas, B[:, j], lw=1.1, color=C_GRAY, alpha=0.7,
                label=rf"$\beta_{{{j+1}}}^*=0$")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"惩罚强度 $\lambda$（左大右小）")
    ax.set_ylabel(r"估计系数 $\widehat{\beta}_j(\lambda)$")
    ax.set_title("Lasso 路径：$\\lambda$ 从大到小，系数逐个「激活」")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, "ch6_lassopath.png")


# ---------------------------------------------------------------- ch7 核方法
def ch8_kernel():
    # XOR 型非线性数据 + 特征映射后线性可分
    n = 80
    t = rng.uniform(0, 2 * np.pi, n)
    r1 = rng.normal(2.0, 0.45, n // 2)
    r2 = rng.normal(4.2, 0.5, n // 2)
    th1 = rng.uniform(0, 2 * np.pi, n // 2)
    th2 = rng.uniform(0, 2 * np.pi, n // 2)
    Z = np.vstack([np.c_[r1 * np.cos(th1), r1 * np.sin(th1)],
                   np.c_[r2 * np.cos(th2), r2 * np.sin(th2)]])
    ylabel = np.array([0] * (n // 2) + [1] * (n // 2))

    Phi = np.c_[Z ** 2, np.sqrt(2) * Z[:, 0] * Z[:, 1]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.scatter(Z[ylabel == 0, 0], Z[ylabel == 0, 1], s=22, color=C_DATA,
               edgecolors="none", alpha=0.7, label="类别 $-1$")
    ax.scatter(Z[ylabel == 1, 0], Z[ylabel == 1, 1], s=22, color=C_FIT,
               edgecolors="none", alpha=0.7, marker="s", label="类别 $+1$")
    ax.set_title("(a) 原始空间 $z$：线性不可分")
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.legend(fontsize=9); ax.set_aspect("equal"); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(Phi[ylabel == 0, 0] + Phi[ylabel == 0, 1],
               Phi[ylabel == 0, 2], s=22, color=C_DATA, edgecolors="none",
               alpha=0.7, label="类别 $-1$")
    ax.scatter(Phi[ylabel == 1, 0] + Phi[ylabel == 1, 1],
               Phi[ylabel == 1, 2], s=22, color=C_FIT, edgecolors="none",
               alpha=0.7, marker="s", label="类别 $+1$")
    ax.set_title(r"(b) 特征空间 $x=\phi(z)$：线性可分")
    ax.set_xlabel("$x_1+x_2=z_1^2+z_2^2$"); ax.set_ylabel(r"$x_3=\sqrt{2}\,z_1z_2$")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "ch8_kernel.png")


# ---------------------------------------------------------------- ch7 double descent
def ch8_doubledescent():
    """真实的 double descent 模拟实验（非示意曲线）：各向同性高斯设计，
    固定 n，让 d 扫过插值阈值 d = n。
    估计量：OLS/最小范数 ridgeless 最小二乘（SVD 形式，第 3、5 章）
    与验证集调参的 ridge（第 3 章）。
    测试 MSE = ||beta_hat - beta||^2 + sigma^2（各向同性测试点下为精确值），
    对 reps 次设计抽样取平均。"""
    n = 200
    sigma = 0.5
    reps = 25
    lam_grid = np.logspace(-1, 3.5, 12)
    dvals = np.unique(np.round(np.concatenate([
        np.geomspace(0.05, 0.95, 40),
        np.linspace(0.96, 1.04, 9),
        np.geomspace(1.06, 8.0, 40),
    ]) * n).astype(int))
    dvals = dvals[dvals >= 2]
    g = dvals / n
    risk_ls = np.zeros(len(dvals))
    risk_rg = np.zeros(len(dvals))
    for i, d in enumerate(dvals):
        acc_ls = np.empty(reps)
        acc_rg = np.empty(reps)
        for r in range(reps):
            X = rng.standard_normal((n, d))
            beta = rng.standard_normal(d)
            beta /= np.linalg.norm(beta)
            y = X @ beta + sigma * rng.standard_normal(n)
            Xv = rng.standard_normal((n, d))
            yv = Xv @ beta + sigma * rng.standard_normal(n)
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            uty = U.T @ y
            # ridgeless：d < n 时为 OLS，d >= n 时为最小范数解（同一 SVD 公式）
            b_ls = Vt.T @ (uty / s)
            acc_ls[r] = np.sum((b_ls - beta) ** 2) + sigma ** 2
            # ridge：lambda 用验证集 (Xv, yv) 上的 MSE 调参
            coef = (s[:, None] / (s[:, None] ** 2 + lam_grid[None, :])) * uty[:, None]
            pred = Xv @ (Vt.T @ coef)                       # n x len(lam_grid)
            mses = np.mean((pred - yv[:, None]) ** 2, axis=0)
            b_rg = Vt.T @ (coef[:, int(np.argmin(mses))])
            acc_rg[r] = np.sum((b_rg - beta) ** 2) + sigma ** 2
        risk_ls[i] = acc_ls.mean()
        risk_rg[i] = acc_rg.mean()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(g, risk_ls, color=C_FIT, lw=2.2,
            label="Ridgeless（$d<n$ 为 OLS，$d\\geq n$ 为最小范数解）：$d=n$ 处尖峰后再降")
    ax.plot(g, risk_rg, color=C_DATA, lw=2.2,
            label="Ridge（验证集调 $\\lambda$）：无尖峰")
    k0 = int(np.argmin(np.abs(g - 1.0)))
    ax.plot([g[k0]], [risk_ls[k0]], "^", color="k", ms=10)
    ax.annotate("插值阈值 $d=n$：\n训练误差恰为零",
                xy=(g[k0], risk_ls[k0]), xytext=(1.7, 120),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="k"))
    ax.axvline(1.0, color=C_GRAY, ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("模型复杂度 $d/n$")
    ax.set_ylabel("测试 MSE，$\\|\\hat{\\beta}-\\beta\\|^2+\\sigma^2$（对数轴）")
    ax.set_title("Double descent 模拟复现（$n=200$，每个 $d$ 取 25 次设计抽样）")
    ax.legend(fontsize=9.5, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    save(fig, "ch8_doubledescent.png")


# ---------------------------------------------------------------- ch7 FDR 控制
def ch7_fdr():
    """(a) BH 几何：排序 p 值 vs 斜线 qk/m 与 Bonferroni 水平线；
    (b) 模拟：信号稀疏度变化时 BH 与 Bonferroni 的功效对比（FDR 均受控）。"""
    from scipy import stats as st

    # ---- (a) BH 几何：一个具体例子
    m = 200
    rng_a = np.random.default_rng(7)
    mu = np.zeros(m)
    mu[:30] = 3.2
    z = mu + rng_a.standard_normal(m)
    p = st.norm.sf(z)                      # 单尾 p 值
    ps = np.sort(p)
    ks = np.arange(1, m + 1)
    q = 0.1
    ok = ps <= q * ks / m
    khat = int(np.max(ks[ok])) if ok.any() else 0

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7))
    ax = axes[0]
    rej = ks <= khat
    ax.plot(ks[~rej], ps[~rej], "o", ms=2.8, color="0.6",
            label="未被拒绝")
    ax.plot(ks[rej], ps[rej], "o", ms=3.2, color=C_FIT,
            label="被 BH 拒绝")
    ax.plot(ks, q * ks / m, "-", color=C_DATA, lw=1.8,
            label="BH 斜线 $qk/m$（$q=0.1$）")
    ax.axhline(q / m, color="0.25", ls="--", lw=1.4,
               label="Bonferroni 水平线 $q/m$")
    if khat:
        ax.plot([khat], [ps[khat - 1]], "s", ms=9, mfc="none",
                mec="k", mew=1.6, label="最后一个交叉点 $\\hat{k}$")
        ax.annotate("最后一个交叉点：\n"
                    f"$p_{'{'}({khat}){'}'}\\leq q\\hat{{k}}/m$，\n"
                    f"拒绝全部 $k\\leq\\hat{{k}}={khat}$，\n"
                    f"其中仅 {(ps[:khat] < q/m).sum()} 个 Bonferroni 也拒绝",
                    xy=(khat, ps[khat - 1]), xytext=(115, 0.40),
                    fontsize=9.5,
                    arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("次序 $k$")
    ax.set_ylabel("排序后 $p$ 值 $p_{(k)}$")
    ax.set_title("(a) BH 的几何：斜线放松阈值，Bonferroni 是水平线")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)

    # ---- (b) 功效模拟：m=1000，信号比例从 0.5% 到 20%，z ~ N(mu,1)
    m2, q2, R = 1000, 0.05, 200
    fracs = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    mu_sig = 3.0
    rng_b = np.random.default_rng(11)

    def bh_k(ps_):
        thr = q2 * np.arange(1, m2 + 1) / m2
        okk = ps_ <= thr
        return int(np.max(np.nonzero(okk)[0]) + 1) if okk.any() else 0

    pow_bh, pow_bf, fdr_bh, fdr_bf = [], [], [], []
    for frac in fracs:
        s = int(round(frac * m2))
        pbh, pbf, fbh, fbf = [], [], [], []
        for _ in range(R):
            zz = rng_b.standard_normal(m2)
            zz[:s] += mu_sig
            pv = 2 * st.norm.sf(np.abs(zz))
            ps_ = np.sort(pv)
            # BH
            k = bh_k(ps_)
            if k > 0:
                tp = int((pv[:s] <= ps_[k - 1]).sum())
                pbh.append(tp / s)
                fbh.append((k - tp) / k)
            else:
                pbh.append(0.0); fbh.append(0.0)
            # Bonferroni
            thr = q2 / m2
            tp = int((pv[:s] <= thr).sum())
            v = int((pv[s:] <= thr).sum())
            r = tp + v
            pbf.append(tp / s)
            fbf.append(v / r if r else 0.0)
        pow_bh.append(np.mean(pbh)); pow_bf.append(np.mean(pbf))
        fdr_bh.append(np.mean(fbh)); fdr_bf.append(np.mean(fbf))

    print("BH   power:", np.round(pow_bh, 3), "FDR:", np.round(fdr_bh, 4))
    print("Bonf power:", np.round(pow_bf, 3), "FDR:", np.round(fdr_bf, 4))

    ax = axes[1]
    ax.plot(fracs * 100, pow_bh, "-o", color=C_FIT, ms=5, lw=2,
            label="BH：功效")
    ax.plot(fracs * 100, pow_bf, "-s", color=C_GRAY, ms=5, lw=2,
            label="Bonferroni：功效")
    ax.fill_between(fracs * 100, pow_bf, pow_bh, color=C_FIT, alpha=0.12)
    ax.annotate(f"稀疏时（信号 $<2\\%$）\nBH 功效约为 Bonferroni 的\n"
                f"{pow_bh[1]/max(pow_bf[1],1e-9):.1f} 倍",
                xy=(fracs[1] * 100, pow_bh[1]),
                xytext=(6.5, 0.45), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=C_FIT))
    ax.set_xscale("log")
    ax.set_xlabel("真实信号比例（%，对数刻度）")
    ax.set_ylabel("功效（真阳性率）")
    ax.set_title("(b) 功效对比（$m=1000$，$q=0.05$，200 次重复；\n"
                 "两种方法的 FDR 均不超过 $q$）")
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "ch7_fdr.png")


if __name__ == "__main__":
    ch1_galton()
    ch2_leastsq()
    ch4_gd()
    ch4_loss()
    ch5_implicit()
    ch6_lassopath()
    ch7_fdr()
    ch8_kernel()
    ch8_doubledescent()
    print("ALL DONE")
