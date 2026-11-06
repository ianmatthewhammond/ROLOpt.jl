# Proof-of-concept evidence deck

A 5-slide deck demonstrating that ROLOpt.jl implements Kouri's inexact proximal
trust-region method (arXiv:2604.07216v1) correctly and faithfully to its Trilinos
ROL `TypeP` blueprint (commit `46b83e0`).

- **Slide 1** — how the algorithm works (TR-Risk loop + dual-SPG prox).
- **Slide 2** — finite-difference Taylor test: `F'(x)` is correct (slope-2
  remainder), plus the package's machine-precision identity checks.
- **Slide 3** — closed-form optima recovered (LASSO = ROL `test_01`, across
  `:cauchy`/`:spg`/`:ncg`; the `(·)₊` kink at `x* = -1/6`) and a precision ladder
  (Float32 / Float64 / BigFloat).
- **Slide 4** — finite-dimensional E+CVaR analogue: support function exact vs. an
  independent water-filling reference; `h_k` driven below the paper's `1e-8`
  target by the improver subsolvers where Cauchy-only stalls.
- **Slide 5** — inexact evaluations honor Conditions 1 & 2 (bounds tighten with
  `h_k`, realized error stays under), convergence to the exact minimizer, and
  provenance.

Everything is generated from problem instances **copied from the passing test
suite** — no invented numbers. `generate_evidence.jl` uses only ROLOpt +
LinearAlgebra + Random + Printf (no plotting dependency); the figures are drawn by
pgfplots from the emitted `data/*.dat`, and tables from `tables/*.tex`.

## Build

```sh
make            # regenerate data/ + tables/, then build slides.pdf (lualatex ×2)
make open       # open slides.pdf
make testcount  # run the full test suite, record the passing count for slide 5
make clean      # remove generated artifacts
```

Requires Julia (`~/.juliaup/bin/julia`) and a LaTeX install with `pgfplots`,
`booktabs`, `siunitx`. On this machine MacTeX lives off-PATH at
`/Library/TeX/texbin/`, so the Makefile defaults to
`LATEX = /Library/TeX/texbin/lualatex`. Override if yours differs, e.g.:

```sh
make LATEX=/usr/local/texlive/2025/bin/universal-darwin/lualatex
make JULIA=/opt/homebrew/bin/julia
```

`tables/testcount.tex` is a "last verified" figure committed alongside the deck
and refreshed only by `make testcount`; the default `make` never runs the suite,
so the deck always builds.

## Honest scope

This is a finite-dimensional proof of concept. The paper's PDE experiments
(Burgers SAA, semilinear elliptic FEM) are **not** reproduced, and there is no
live Trilinos ROL C++ comparison. The match to the paper is argued through theory
(the `h_k` stationarity criterion and Conditions 1 & 2), closed-form / ROL-bridge
optima, metric/precision genericity, and line-by-line provenance.
