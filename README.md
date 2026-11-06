# ROLOpt.jl

> [!WARNING]
> **Unofficial, experimental, and not fully working.** ROLOpt.jl is an
> independent, unofficial Julia port. It is **not** affiliated with, reviewed
> by, or endorsed by Sandia National Laboratories, the Trilinos Project, the ROL
> developers, or the authors of the papers it implements. It is a research
> prototype: it has been tested on finite-dimensional problems only, it has
> known open issues (see [Known limitations](#known-limitations)), and its API
> may change without notice. Do not rely on it for production work. For the
> real thing, use [ROL](https://rol.sandia.gov) itself.

A Julia implementation of the inexact proximal trust-region method for
structured nonsmooth / risk-averse optimization from

> D. P. Kouri, *An Inexact Trust-Region Method for Structured Nonsmooth
> Optimization with Application to Risk-Averse Stochastic Programming*,
> arXiv:2604.07216 (2026). <https://arxiv.org/abs/2604.07216>

This port follows **v1** of the paper (April 2026). The paper has since been
revised (v4, October 2026), and this code has not been checked against the
revisions.

with its trust-region scaffold ported from the Rapid Optimization Library
([ROL](https://rol.sandia.gov)), part of the
[Trilinos Project](https://trilinos.github.io). Please cite those works, not
this package, if you use the method (see [Citing](#citing)).

It solves

```
min_x  J(x) = f(x) + suppA(F(x)) + phi(x)
```

where `f` is smooth, `F: X → Y` is smooth, `phi` is proper closed convex
(prox-friendly), and `suppA` is the support function of a bounded closed convex
risk envelope `A ⊆ Y`. The method (paper Algorithm 1, "TR-Risk") builds, each
iteration, a quadratic model of `f` plus a re-anchored nonsmooth term
`psi_k(x) = suppA(A_k(x - x_k) + b_k) + phi(x)`, and computes the prox of
`psi_k` by a nonmonotone spectral projected gradient on the dual (paper
Algorithm 2).

The trust-region scaffold ports Trilinos ROL's
`TypeP::TrustRegionAlgorithm` (the `baraldi2022proximal` / `baraldi2024efficient`
solver), pinned to Trilinos commit `46b83e0`; each source file cites the C++ it
mirrors. The net-new content is the risk generalization: the `psi_k` objective,
Algorithm 2, and the `RiskEnvelope` abstraction.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| **1** | Cauchy-only TR-Risk, exact evaluations, Euclidean spaces; Algorithm 2; `IntervalEnvelope`; Zero/L1/Bound/BoxL1 `phi` | **done, tested** |
| **2a** | Diagonal weighted (mass-matrix / probability) spaces, metric-aware prox/adjoint, `ConvexCombinationEnvelope` (E+CVaR) | **done, tested** |
| **2b-i** | SPG improver subsolver (`subsolver=:spg`) — makes the nonsmooth CVaR problem converge fast | **done, tested** |
| **2b-ii** | Wired inexactness (Conditions 1–2) via `InexactOracle` + `cred`, audited against the post-model `h_k` | **done, tested** |
| **2b-iii** | `(theta_k, F'')` Lagrangian curvature term in `B_k` (`use_curvature=true`), `theta_k` frozen from the stationarity prox | **done, tested** |
| **3** | Hardening: zero-alloc hot kernels + `@allocated`/`@inferred` guards, function-barrier `solve` (type-stable loop), generic precision (BigFloat / Float32), NCG subsolver (`subsolver=:ncg`) | **done, tested** |

Weighted spaces use a **diagonal** metric (`M = diag(w)`): element volumes for the
control space `X`, probabilities for `Y = L²(Ω,P)`. This is what the paper uses
for mesh independence and keeps the separable prox / box projection closed-form.
Convergence speed tracks metric/Hessian compatibility — with the natural pairing
(`f = (τ/2)‖x‖_X²`) the weighted solve converges as fast as the Euclidean one.

## Example

```julia
using ROLOpt

# min  ½‖x‖²  +  max(0, c·x + 0.5)        (A = [0,1], so suppA = (·)₊)
n = 3
f   = TikhonovObjective(1.0)
F   = AffineMap(reshape([1.0, 1.0, 1.0], 1, n), [0.5])   # scalar-valued F
phi = ZeroProx{Float64}()
A   = IntervalEnvelope(0.0, 1.0)

prob = RiskProblem(f, F, phi, A)
res  = solve(prob, zeros(n); opts = TROptions{Float64}(verbose = 1))

res.x          # ≈ [-1/6, -1/6, -1/6]
res.J, res.hk  # objective value and stationarity metric
```

## Building blocks

- `HilbertSpace` / `EuclideanSpace` / `WeightedSpace` — inner product + Riesz map.
- `SmoothObjective`: `DiagonalQuadratic`, `QuadraticObjective`, `TikhonovObjective`, `ZeroObjective`.
- `VectorMap` (`F`): `AffineMap`, `QuadraticMap` (explicit); `OperatorMap` (pure
  matrix-free — user JVP/VJP callbacks, no stored Jacobian); `CachedJacobianMap`
  (nonlinear `F`, `A_k = F'(x_k)` assembled once per anchor via a `prepare!` hook).
  You choose the representation; ROLOpt never auto-assembles `A_k`.
- `ProxObjective` (`phi`): `L1Objective`, `BoundConstraint`, `BoxL1Objective`, `ZeroProx`.
- `RiskEnvelope`: `IntervalEnvelope` (box), `ConvexCombinationEnvelope` (E + CVaR), `SimplexEnvelope` (probability simplex, so `suppA = max`) — metric-aware support function + projection; optional `warmdual!` dual warm start.
- `PsiModel` + `prox!` — Algorithm 2 (returns an isbits `DualSPGResult` status; the optimal dual is left in `psi.theta`).
- `RiskProblem`, `solve`, `TROptions` — the TR-Risk driver. Trial step `opts.subsolver` ∈ `:cauchy` / `:spg` / `:ncg`; optional `(theta_k, F'')` curvature (`use_curvature=true`); inexact evaluations via an `InexactOracle` (Conditions 1 & 2). Generic over `T<:Real` (Float64 / Float32 / BigFloat).
- `check_gradient` / `check_hessvec` / `check_jacobian` / `check_adjoint` / `check_hess_sym` — finite-difference derivative checks.

## Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```

## Known limitations

This is a proof of concept, not a finished solver. In particular:

- **Finite-dimensional only.** Neither of the paper's PDE examples (risk-averse
  control of Burgers' equation, sparse semilinear elliptic control) is
  implemented, and no table from the paper has been reproduced. Agreement with
  the paper is argued from closed-form optima, finite-difference checks,
  stationarity certificates and the inexactness conditions, not from a
  side-by-side run of ROL's C++.
- **Hard-max problems can stall.** With `SimplexEnvelope` (a true sampled
  maximum) and an ill-conditioned matrix-free Jacobian, the inner dual SPG prox
  (Algorithm 2) can plateau before reaching its tolerance, which then forces the
  trust region to reject good steps. This is an open problem in this port.
- **Known defect in the dual line search.** In `prox!` (`src/psimodel.jl`), if
  backtracking reaches `lsmax` without satisfying the Armijo condition, the
  trial dual step is still accepted.
- **Not ported from ROL:** the TR-radius prox root-find (`dprox`), and anything
  beyond diagonal (mass-matrix) metrics.
- No package registration, no stable API, no performance tuning beyond the
  zero-allocation hot kernels.

Issues and corrections are welcome.

## Citing

If this code is useful to you, please cite the original method and software:

```bibtex
@misc{kouri2026inexact,
  author        = {Kouri, Drew P.},
  title         = {An Inexact Trust-Region Method for Structured Nonsmooth
                   Optimization with Application to Risk-Averse Stochastic Programming},
  year          = {2026},
  eprint        = {2604.07216},
  archivePrefix = {arXiv}
}

@article{baraldi2022proximal,
  author  = {Baraldi, Robert J. and Kouri, Drew P.},
  title   = {A proximal trust-region method for nonsmooth optimization with
             inexact function and gradient evaluations},
  journal = {Mathematical Programming},
  volume  = {201},
  number  = {1},
  pages   = {1--40},
  year    = {2022}
}

@article{baraldi2024efficient,
  author  = {Baraldi, Robert J. and Kouri, Drew P.},
  title   = {Efficient Subproblem Solvers for a Nonsmooth Trust-Region Method},
  journal = {Computational Optimization and Applications},
  year    = {2024}
}
```

and ROL / Trilinos (<https://rol.sandia.gov>, <https://trilinos.github.io>).
The paper itself is not redistributed here; get it from arXiv.

## License

ROLOpt.jl is released under the MIT License (see [`LICENSE`](LICENSE)).
The portions ported from ROL remain subject to ROL's BSD-3-Clause license (Copyright (c) 2014 NTESS and the ROL contributors),
reproduced in [`LICENSE-ROL`](LICENSE-ROL). Use of the names Sandia, Trilinos
or ROL here is for attribution only and implies no endorsement.
