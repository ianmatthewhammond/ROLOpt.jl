# ROLOpt.jl

A Julia implementation of the inexact proximal trust-region method for
structured nonsmooth / risk-averse optimization from

> D. P. Kouri, *An Inexact Trust-Region Method for Structured Nonsmooth
> Optimization with Application to Risk-Averse Stochastic Programming*
> (arXiv 2604.07216v1; see `paper/`).

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

The trust-region scaffold faithfully ports Trilinos ROL's
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
| 2b | Wired inexactness (Conditions 1–2), `(theta_k, F'')` curvature, NCG subsolver | planned |
| 3 | Performance / type-stability hardening (zero-alloc hot loops, BigFloat) | planned |

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
- `VectorMap` (`F`): `AffineMap`, `QuadraticMap`.
- `ProxObjective` (`phi`): `L1Objective`, `BoundConstraint`, `BoxL1Objective`, `ZeroProx`.
- `RiskEnvelope`: `IntervalEnvelope` (support function + projection).
- `PsiModel` + `prox!` — Algorithm 2 (returns a `DualSPGResult` with convergence diagnostics).
- `RiskProblem`, `solve`, `TROptions` — the TR-Risk driver.
- `check_gradient` / `check_hessvec` / `check_jacobian` / `check_adjoint` / `check_hess_sym` — finite-difference derivative checks.

## Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```
