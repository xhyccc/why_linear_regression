#!/usr/bin/env python3
"""Generate all figures for "Why (we still need) linear regression" (English edition).
Unified style: managed runtime setup_plot, fixed random seed, vector PDF output to figures/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

setup_plot()
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
rng = np.random.default_rng(20261001)

# Unified palette (restrained academic style)
C_DATA = "#4C72B0"
C_FIT = "#C44E52"
C_AUX = "#55A868"
C_GRAY = "#7F7F7F"


def save(fig, name):
    name = name.replace(".png", ".pdf")   # vector output
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- ch1 Galton
def ch1_galton():
    n = 300
    x = rng.normal(172, 6, n)                     # father height, cm
    r = 0.5
    y = 172 + r * (x - 172) + rng.normal(0, 5.2, n)  # son height, regression to the mean
    b = r * y.std() / x.std()
    a = y.mean() - b * x.mean()
    xs = np.linspace(150, 196, 10)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(x, y, s=14, alpha=0.35, color=C_DATA, edgecolors="none",
               label="Father–son heights (simulated)")
    ax.plot(xs, xs, "--", color=C_GRAY, lw=1.4,
            label=r"$y=x$ (perfect inheritance)")
    ax.plot(xs, a + b * xs, "-", color=C_FIT, lw=2.2,
            label=rf"Regression line $\hat{{y}}={a:.1f}+{b:.2f}\,x$")
    ax.annotate("Slope $=r\\,\\sigma_y/\\sigma_x<1$:\ntall fathers have sons\nwho are on average shorter",
                xy=(184, a + b * 184), xytext=(188, 158),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_FIT, lw=1.2),
                color=C_FIT)
    ax.set_xlabel("Father's height $x$ (cm)")
    ax.set_ylabel("Son's height $y$ (cm)")
    ax.set_title("Galton's regression to the mean: the regression line is flatter than $y=x$")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(alpha=0.25)
    save(fig, "ch1_galton.png")


# ---------------------------------------------------------------- ch2 triptych
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
    ax.plot(xs, np.polyval(b_hat, xs), color=C_FIT, lw=2.2, label=r"Least-squares line")
    for xi, yi in list(zip(x, y))[::6]:
        ax.plot([xi, xi], [yi, np.polyval(b_hat, xi)], color=C_GRAY, lw=0.9,
                alpha=0.8)
    ax.set_title("(a) Fitting: minimum sum of squared residuals")
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
    ax.set_title("(b) Objective: unimodal and tractable")
    ax.set_xlabel("Slope $b$"); ax.set_ylabel("Sum of squared residuals")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.annotate("", xy=(7.2, 8.2), xytext=(0.6, 0.8),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))
    ax.text(7.0, 9.0, r"$\mathbf{y}\in\mathbb{R}^n$", fontsize=13)
    ax.annotate("", xy=(6.4, 1.7), xytext=(0.6, 0.8),
                arrowprops=dict(arrowstyle="-|>", color=C_FIT, lw=1.6))
    ax.text(5.6, 0.3, r"$\mathbf{x}\,b$ (one-dimensional subspace)", fontsize=12, color=C_FIT)
    ax.annotate("", xy=(6.4, 1.7), xytext=(7.2, 8.2),
                arrowprops=dict(arrowstyle="-|>", color=C_AUX, lw=1.6))
    ax.text(7.8, 4.6, r"residual $\perp$ subspace", fontsize=11, color=C_AUX)
    ax.plot([6.4], [1.7], "o", color="k", ms=6)
    ax.text(6.5, 2.6, r"$\mathbf{x}\hat{b}$ (projection)", fontsize=11)
    ax.set_title("(c) Geometry: least squares $=$ orthogonal projection")
    fig.tight_layout()
    save(fig, "ch2_leastsq.png")


# ---------------------------------------------------------------- ch4 GD
def ch4_gd():
    # Quadratic objective L(b)=1/2 b^T H b, eigenvalues of H: 1 and 0.04 (condition number 25)
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
        ax.plot(0, 0, "k*", ms=11, label="Least-squares solution $\\widehat{\\beta}$")

    b0 = (3.5, 2.6)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7))

    def style(ax, title, legend_loc="upper left"):
        ax.set_xlim(-4.6, 4.6); ax.set_ylim(-3.4, 4.0)
        ax.set_title(title)
        ax.set_xlabel("$\\beta_1$")
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc=legend_loc)

    # (a) Monotone: eta=0.9, trajectory stays on one side of the valley
    ax = axes[0]
    contours(ax)
    tr = run_gd(0.9, 60, b0)
    ax.plot(tr[:, 0], tr[:, 1], "-o", color=C_AUX, ms=3.5, lw=1.5,
            markevery=6, label="$\\eta=0.9$: same-sign approach")
    ax.plot([b0[0]], [b0[1]], "o", color="k", ms=6, mfc="none",
            label="Initialization $\\widehat{\\beta}_0$")
    ax.annotate("Trajectory stays on one side of the valley:\nthe error projection keeps its sign; slow but steady",
                xy=(-0.5, 0.8), xytext=(-4.35, 3.25), fontsize=10,
                color=C_AUX, arrowprops=dict(arrowstyle="->", color=C_AUX))
    style(ax, "(a) Monotone: $0<\\eta\\leq 1/\\lambda_{\\max}$")
    ax.set_ylabel("$\\beta_2$")

    # (b) Oscillating convergence: eta=1.5, each step crosses the valley center line
    ax = axes[1]
    contours(ax)
    tr = run_gd(1.5, 30, b0)
    ax.plot(tr[:, 0], tr[:, 1], "-o", color=C_FIT, ms=4, lw=1.7,
            markevery=1, label="$\\eta=1.5$: zigzag convergence")
    for k, off in [(0, (8, 2)), (1, (-14, 8)), (2, (8, -2)), (3, (-16, -12))]:
        ax.annotate(f"$t={k}$", xy=(tr[k, 0], tr[k, 1]),
                    xytext=off, textcoords="offset points",
                    fontsize=8.5, color=C_FIT)
    ax.annotate("Each step crosses the valley center line and flips sign,\nwith amplitude decaying at $|1-\\eta\\lambda_{\\max}|=0.5$",
                xy=(tr[2, 0], tr[2, 1]), xytext=(-4.35, -2.9), fontsize=10,
                color=C_FIT, arrowprops=dict(arrowstyle="->", color=C_FIT))
    style(ax, "(b) Oscillating convergence: $1/\\lambda_{\\max}<\\eta<2/\\lambda_{\\max}$")

    # (c) Oscillating divergence: eta=2.6, sign flips with growing amplitude
    ax = axes[2]
    contours(ax)
    b0d = (0.9, 0.7)          # small initialization: first steps stay in view
    tr = run_gd(2.6, 7, b0d)   # eta*lambda_max>2, sign flips, amplitude x|1-2.6|=1.6
    ax.plot(tr[:, 0], tr[:, 1], "-o", color="#8172B3", ms=4.5, lw=1.8,
            label="$\\eta=2.6$: diverging zigzag")
    for k in range(len(tr)):
        ax.annotate(f"$t={k}$", xy=(tr[k, 0], tr[k, 1]),
                    xytext=(6, 5), textcoords="offset points", fontsize=8.5,
                    color="#8172B3")
    ax.annotate("Each step flips sign, amplitude grows by $|1-\\eta\\lambda_{\\max}|=1.6$\nand leaves the view within a few steps",
                xy=(tr[-1, 0], tr[-1, 1]), xytext=(-4.35, -2.9), fontsize=10,
                color="#8172B3",
                arrowprops=dict(arrowstyle="->", color="#8172B3"))
    style(ax, "(c) Oscillating divergence: $\\eta\\geq 2/\\lambda_{\\max}$")

    fig.tight_layout()
    save(fig, "ch4_gd.png")


# ---------------------------------------------------------------- ch4 signed error vs loss
def ch4_loss():
    """Three regimes of eta: signed error (left axis, linear) vs loss (right axis, log).

    Key fact: the loss is a weighted sum of even powers of the per-direction errors,
    so it never oscillates; oscillation shows up only in the sign of the trajectory
    (signed error).
    """
    A = np.array([[1.0, 0], [0, 0.2]])
    R = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    H = R @ (A ** 2) @ R.T
    _, V = np.linalg.eigh(H)
    vmax = V[:, -1]              # top eigenvalue direction
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
        (0.5, 40, "Monotone decrease",
         "$\\eta\\,\\lambda_{\\max}=0.5\\leq 1$:\nthe error keeps its sign;\nthe loss decreases monotonically",
         C_AUX, None),
        (1.5, 40, "Oscillating decrease",
         "$1<\\eta\\,\\lambda_{\\max}=1.5<2$:\nthe signed error flips each step (sawtooth);\n"
         "but the loss (gray dashed) stays monotone—\neven powers kill the sign",
         C_FIT, None),
        (2.6, 12, "Oscillating divergence",
         "$\\eta\\,\\lambda_{\\max}=2.6>2$:\neach step flips sign, amplitude\ngrows by $|1-\\eta\\lambda_{\\max}|=1.6$",
         "#8172B3", (-1500, 1500)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    for k, (ax, (eta, T, name, lab, c, ylim)) in enumerate(zip(axes, cases)):
        ss, ls = series(eta, T)
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        l1, = ax.plot(ss, "-o", color=c, ms=3.5, lw=1.8,
                      markevery=max(1, len(ss) // 16),
                      label="Signed error $v_{\\max}^{\\top}(\\widehat{\\boldsymbol{\\beta}}_t-\\widehat{\\boldsymbol{\\beta}})$")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"({'abc'[k]}) {name}: $\\eta={eta}$", fontsize=12)
        ax.set_xlabel("Iteration $t$")
        ax.grid(alpha=0.3)
        axr = ax.twinx()
        l2, = axr.semilogy(ls, "--", color="0.45", lw=1.3,
                           label="Loss (log scale)")
        axr.tick_params(axis="y", labelsize=8, colors="0.35")
        if k == 2:
            axr.set_ylabel("Loss $L(\\widehat{\\boldsymbol{\\beta}}_t)-L_{\\min}$ (log)",
                           fontsize=9, color="0.35")
        ax.legend(handles=[l1, l2], fontsize=8, loc="upper right",
                  framealpha=0.9)
        ax.text(0.04, 0.04, lab, transform=ax.transAxes, fontsize=9,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=c, alpha=0.92))
    axes[0].set_ylabel("Signed error (linear scale)")
    fig.tight_layout()
    save(fig, "ch4_loss.png")


# ---------------------------------------------------------------- ch5 implicit regularization
def ch5_implicit():
    n, d = 30, 100
    X = rng.normal(size=(n, d))
    beta_true = np.zeros(d)
    beta_true[:5] = [3, -2, 1.5, 1, -0.8]
    y = X @ beta_true

    # Spectral decay + GD limit curve
    sv = np.linalg.svd(X, compute_uv=False)
    lam = sv ** 2
    kappa = np.linspace(0, 1.5, 200)  # (kept: spare grid)
    # Analytic: beta_hat(kappa) = V diag(1-(1-kappa*lam)^T) V^T X^T y, T=10000 approximates the fixed point
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
           label="True $\\beta^*$ (5 nonzero)")
    ax.bar(np.arange(12) + 0.2, beta_gd[idx], width=0.4, color=C_FIT,
           label="GD limit $\\widehat{\\beta}_{\\mathrm{GD}}$ (zero initialization)")
    ax.set_xticks(np.arange(12), [f"$\\beta_{{{i+1}}}$" for i in range(12)],
                  fontsize=9)
    ax.set_title("(a) GD automatically finds the 'energy-concentrated' interpolating solution")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    # Spectral filtering view: g(lambda)=1-(1-eta*lambda)^T for different iteration counts T
    lv = np.linspace(0, 1, 300)  # lambda/lambda_max
    for T, c in [(50, C_GRAY), (400, C_AUX), (4000, C_FIT)]:
        g = 1 - (1 - lv) ** T
        ax.plot(lv, g, lw=2.0, color=c, label=rf"Iterations $T={T}$")
    ax.text(0.35, 0.55, r"$g(\lambda)=1-(1-\eta\lambda)^T$", fontsize=11)
    ax.set_xlabel("Relative eigenvalue $\\lambda/\\lambda_{\\max}$")
    ax.set_ylabel("Spectral filter $g(\\lambda)$")
    ax.set_ylim(-0.03, 1.15)
    ax.set_title("(b) GD limit $=$ a 'low-pass filter' in the spectral domain")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "ch5_implicit.png")


# ---------------------------------------------------------------- ch6 lasso path
def ch6_lassopath():
    n, d = 60, 8
    X = rng.normal(size=(n, d)) / np.sqrt(n)   # unit-scale columns, X_j^T X_j / n ~ 1
    beta = np.array([4, -3, 2.5, 0, 0, 1.5, 0, 0.0])
    y = X @ beta + rng.normal(0, 0.3, n)

    # Coordinate descent for the lasso path: min (1/2n)||y-Xb||^2 + lambda||b||_1
    # Update: b_j = S(X_j^T r / n, lambda) / (X_j^T X_j / n)
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
                label=rf"$\beta_{{{j+1}}}^*={beta[j]}$ (truly nonzero)")
    for j in zero:
        ax.plot(lambdas, B[:, j], lw=1.1, color=C_GRAY, alpha=0.7,
                label=rf"$\beta_{{{j+1}}}^*=0$")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"Penalty strength $\lambda$ (large $\to$ small, left to right)")
    ax.set_ylabel(r"Estimated coefficient $\widehat{\beta}_j(\lambda)$")
    ax.set_title("Lasso path: as $\\lambda$ decreases, coefficients are 'activated' one by one")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, "ch6_lassopath.png")


# ---------------------------------------------------------------- ch7 kernel methods
def ch7_kernel():
    # XOR-like nonlinear data + linear separability after feature mapping
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
               edgecolors="none", alpha=0.7, label="Class $-1$")
    ax.scatter(Z[ylabel == 1, 0], Z[ylabel == 1, 1], s=22, color=C_FIT,
               edgecolors="none", alpha=0.7, marker="s", label="Class $+1$")
    ax.set_title("(a) Original space $z$: not linearly separable")
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.legend(fontsize=9); ax.set_aspect("equal"); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(Phi[ylabel == 0, 0] + Phi[ylabel == 0, 1],
               Phi[ylabel == 0, 2], s=22, color=C_DATA, edgecolors="none",
               alpha=0.7, label="Class $-1$")
    ax.scatter(Phi[ylabel == 1, 0] + Phi[ylabel == 1, 1],
               Phi[ylabel == 1, 2], s=22, color=C_FIT, edgecolors="none",
               alpha=0.7, marker="s", label="Class $+1$")
    ax.set_title(r"(b) Feature space $x=\phi(z)$: linearly separable")
    ax.set_xlabel("$x_1+x_2=z_1^2+z_2^2$"); ax.set_ylabel(r"$x_3=\sqrt{2}\,z_1z_2$")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "ch7_kernel.png")


# ---------------------------------------------------------------- ch7 double descent
def ch7_doubledescent():
    g = np.geomspace(0.05, 1.0, 120)
    g2 = np.geomspace(1.0, 30, 120)
    # Classical U-shape + interpolation spike + second descent (schematic); the two pieces meet at the threshold
    u = 0.16 + 0.5 * np.exp(-2.5 * g) + 0.24 * g ** 3
    peak = 0.9
    dd = 0.16 + 0.75 / np.sqrt(g2) + 0.02 * np.log(g2)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(g, u, color=C_DATA, lw=2.2, label="Classical bias–variance trade-off")
    ax.plot([g[-1], 1.0], [u[-1], peak], color=C_DATA, lw=2.2)
    ax.plot(g2, dd, color=C_FIT, lw=2.2,
            label="Modern observation: error decreases again past the interpolation threshold (double descent)")
    ax.plot([1.0], [peak], "^", color="k", ms=10)
    ax.annotate("Interpolation threshold $d=n$:\ntraining error is exactly zero",
                xy=(1.0, peak), xytext=(3.2, 0.75), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.axvline(1.0, color=C_GRAY, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Model complexity (e.g., $d/n$)")
    ax.set_ylabel("Test error")
    ax.set_title("Double descent: overparameterized interpolating solutions are not necessarily bad")
    ax.legend(fontsize=9.5, loc="lower left")
    ax.grid(alpha=0.25, which="both")
    save(fig, "ch7_doubledescent.png")


# ---------------------------------------------------------------- ch7 FDR control
def ch7_fdr():
    """(a) BH geometry: sorted p-values vs the sloped line qk/m and the Bonferroni level;
    (b) simulation: power of BH vs Bonferroni as signal sparsity varies (FDR controlled for both)."""
    from scipy import stats as st

    # ---- (a) BH geometry: a concrete example
    m = 200
    rng_a = np.random.default_rng(7)
    mu = np.zeros(m)
    mu[:30] = 3.2
    z = mu + rng_a.standard_normal(m)
    p = st.norm.sf(z)                      # one-sided p-values
    ps = np.sort(p)
    ks = np.arange(1, m + 1)
    q = 0.1
    ok = ps <= q * ks / m
    khat = int(np.max(ks[ok])) if ok.any() else 0

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7))
    ax = axes[0]
    rej = ks <= khat
    ax.plot(ks[~rej], ps[~rej], "o", ms=2.8, color="0.6",
            label="Not rejected")
    ax.plot(ks[rej], ps[rej], "o", ms=3.2, color=C_FIT,
            label="Rejected by BH")
    ax.plot(ks, q * ks / m, "-", color=C_DATA, lw=1.8,
            label="BH line $qk/m$ ($q=0.1$)")
    ax.axhline(q / m, color="0.25", ls="--", lw=1.4,
               label="Bonferroni level $q/m$")
    if khat:
        ax.plot([khat], [ps[khat - 1]], "s", ms=9, mfc="none",
                mec="k", mew=1.6, label="Last crossing $\\hat{k}$")
        ax.annotate("Last crossing:\n"
                    f"$p_{'{'}({khat}){'}'}\\leq q\\hat{{k}}/m$,\n"
                    f"reject all $k\\leq\\hat{{k}}={khat}$,\n"
                    f"of which only {(ps[:khat] < q/m).sum()} are also rejected by Bonferroni",
                    xy=(khat, ps[khat - 1]), xytext=(115, 0.40),
                    fontsize=9.5,
                    arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("Sorted $p$-values $p_{(k)}$")
    ax.set_title("(a) BH geometry: the sloped line relaxes the threshold; Bonferroni is a horizontal line")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)

    # ---- (b) Power simulation: m=1000, signal fraction from 0.5% to 20%, z ~ N(mu,1)
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
            label="BH: power")
    ax.plot(fracs * 100, pow_bf, "-s", color=C_GRAY, ms=5, lw=2,
            label="Bonferroni: power")
    ax.fill_between(fracs * 100, pow_bf, pow_bh, color=C_FIT, alpha=0.12)
    ax.annotate(f"In the sparse regime (signal $<2\\%$)\nBH power is about "
                f"{pow_bh[1]/max(pow_bf[1],1e-9):.1f}$\\times$\nthat of Bonferroni",
                xy=(fracs[1] * 100, pow_bh[1]),
                xytext=(6.5, 0.45), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=C_FIT))
    ax.set_xscale("log")
    ax.set_xlabel("True signal proportion (%, log scale)")
    ax.set_ylabel("Power (true positive rate)")
    ax.set_title("(b) Power comparison ($m=1000$, $q=0.05$, 200 replications;\n"
                 "FDR of both methods stays below $q$)")
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
    ch7_kernel()
    ch7_doubledescent()
    print("ALL DONE")
