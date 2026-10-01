# Why Linear Regression

A first-principles tutorial on linear regression, by 熊昊一 (Haoyi Xiong).

## Build

```bash
latexmk -xelatex main.tex      # or: xelatex main.tex (twice, for TOC/refs)
```

The document requires XeLaTeX (for Chinese characters in the author name) and
standard TeX Live packages (ctex, amsmath, tcolorbox, tikz, hyperref).

## Structure

- `main.tex` — preamble, title, table of contents
- `chapters/01_intro.tex` — the question, setup, notation
- `chapters/02_least_squares.tex` — deriving the OLS estimator
- `chapters/03_why_squared.tex` — interrogating the loss function
- `chapters/04_geometry.tex` — regression as orthogonal projection
- `chapters/05_gauss_markov.tex` — the BLUE theorem and its boundaries
- `chapters/06_probability.tex` — distributions, t/F tests, intervals
- `chapters/07_bias_variance.tex` — ridge, lasso, cross-validation
- `chapters/08_beyond.tex` — features, kernels, GLMs, neural nets
- `chapters/09_practice.tex` — diagnostics and assumption failures
- `chapters/10_epilogue.tex` — the seven answers, collected
- `chapters/a1_linear_algebra.tex` — appendix of linear algebra facts
- `chapters/a2_exercises.tex` — exercises with full solutions
