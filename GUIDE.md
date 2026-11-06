# ROLOpt.jl — A Complete Walkthrough Guide

> Read this top to bottom. By the end you will understand every file, know what
> was ported from Trilinos ROL versus written new, be able to set up and solve a
> problem, and be able to defend any design decision as if you wrote it yourself.
>
> Paper pointers throughout refer to the LaTeX source of arXiv:2604.07216v1
> (`main.tex`, downloadable from arXiv; not redistributed here) by line
> number, written as `main.tex:NNN`. Code pointers are `file.jl:NNN`.

---

## Table of contents

1. [Part I — The problem and the idea](#part-i--the-problem-and-the-idea)
2. [Part II — Setup and your first solve](#part-ii--setup-and-your-first-solve)
3. [Part III — The mental model you must hold](#part-iii--the-mental-model-you-must-hold)
4. [Part IV — File-by-file walkthrough](#part-iv--file-by-file-walkthrough)
5. [Part V — One `solve` call, traced end to end](#part-v--one-solve-call-traced-end-to-end)
6. [Part VI — The math-to-code map](#part-vi--the-math-to-code-map)
7. [Part VII — Provenance ledger: ROL vs new](#part-vii--provenance-ledger-rol-vs-new)
8. [Part VIII — The three advanced axes](#part-viii--the-three-advanced-axes)
9. [Part IX — The test suite, file by file](#part-ix--the-test-suite-file-by-file)
10. [Part X — The demo evidence pipeline](#part-x--the-demo-evidence-pipeline)
11. [Part XI — Convergence theory you should be able to recite](#part-xi--convergence-theory-you-should-be-able-to-recite)
12. [Part XII — Honest scope: the paper's numerics vs what we built](#part-xii--honest-scope-the-papers-numerics-vs-what-we-built)
13. [Part XIII — Glossary: symbol ↔ code name](#part-xiii--glossary-symbol--code-name)
14. [Part XIV — "Pretend you wrote it": defense Q&A](#part-xiv--pretend-you-wrote-it-defense-qa)
15. [Part XV — Where to go next](#part-xv--where-to-go-next)

---

## Part I — The problem and the idea

### I.1 The one problem this package solves

Everything in this repo exists to solve a single optimization problem
(`main.tex:151`, eq. `optprob`):

```
min_x  J(x) = f(x) + suppA(F(x)) + phi(x)
```

over a real Hilbert space `X`, where:

- **`f : X → ℝ`** is **smooth** (Fréchet differentiable, Lipschitz gradient).
  Think: a Tikhonov regularizer `(τ/2)‖x‖²`, a quadratic, a misfit term.
- **`F : X → Y`** is a **smooth map** into a second Hilbert space `Y`. In the
  risk setting, `F(x)` is a *random variable* — its components are outcomes
  across scenarios.
- **`phi : X → (−∞, +∞]`** is **proper, closed, convex, and prox-friendly**: you
  can evaluate its proximity operator cheaply. Think: an L1 penalty, a box
  indicator, or both.
- **`suppA`** is the **support function** of a risk envelope
  `A ⊆ Y` (nonempty, closed, convex, bounded):
  `suppA(y) = sup_{θ∈A} (θ, y)_Y`. This is the nonsmooth, "risk" term.

The objective mixes three difficulties: a smooth-but-nonconvex part (`f∘`
through `F`), a nonsmooth convex penalty (`phi`), and a nonsmooth term that is
itself a supremum (`suppA∘F`). Standard smooth solvers can't touch it; standard
nonsmooth solvers choke when `F` is expensive (a PDE solve per evaluation).

### I.2 Why "suppA(F(x))" is risk-averse optimization

Here is the connection that makes the whole thing click (`main.tex:164`, eq.
`riskmin`). A **coherent risk measure** `R` has a dual representation as a
support function over its **risk envelope** `A`:

```
R(F(x)) = sup_{θ∈A} (θ, F(x))_Y = suppA(F(x)).
```

So minimizing `f + R(F(x)) + phi` — "minimize a regularized, risk-adjusted
expected loss" — *is exactly* problem `optprob`. The envelope `A` encodes which
risk measure you chose:

- `A = {θ : E[θ]=1, θ≥0}` ⟹ `R = E` (expectation; risk-neutral).
- `A = {θ : E[θ]=1, 0 ≤ θ ≤ 1/(1−p)}` ⟹ `R = AVaR_p` (average value-at-risk,
  a.k.a. CVaR / expected shortfall / superquantile).
- A convex combination ⟹ `R = (1−λ)·E + λ·AVaR_p`, the mean-risk blend the
  paper uses (`main.tex:1505`).

In this repo those envelopes are the `IntervalEnvelope` and
`ConvexCombinationEnvelope` types in `src/riskenvelope.jl`.

### I.3 The strategy: a trust region with a clever inner solve

The method is a **trust-region** algorithm. Each outer iteration:

1. Builds a **local model** `m_k` of `J` around the current iterate `x_k`: a
   quadratic for the smooth part plus a *re-anchored* version of the nonsmooth
   part, `psi_k` (`main.tex:496`).
2. Approximately minimizes `m_k` inside a ball of radius `Δ_k` (the trust
   region) to get a trial step.
3. Accepts or rejects the step by comparing actual vs. predicted decrease, and
   grows/shrinks `Δ_k` accordingly.

The twist: the nonsmooth model term

```
psi_k(x) = suppA( A_k(x − x_k) + b_k ) + phi(x)        (main.tex:380)
```

does **not** have a closed-form proximity operator (it's a support function
composed with an affine map, plus `phi`). So there is a *second, inner*
algorithm whose only job is to compute `prox_{t·psi_k}` — by running a
nonmonotone spectral projected-gradient method on the **dual** problem
(`main.tex:687`).

That is the heart of the whole thing: **two nested algorithms.**

| | Paper | Code |
|---|---|---|
| **Outer** | Algorithm 1, "TR-Risk" (`main.tex:591`) | `src/trustregion.jl` |
| **Inner** | Algorithm 2, "ψₖ prox" (`main.tex:687`) | `src/psimodel.jl`, function `prox!` |

### I.4 The ROL lineage in one paragraph

The outer trust-region machinery is **not new** — it is the proximal trust-region
method of Baraldi & Kouri, shipped in Trilinos ROL as
`ROL::TypeP::TrustRegionAlgorithm` (the `baraldi2022proximal` /
`baraldi2024efficient` solver), pinned here to commit `46b83e0`. That solver
handles `min f(x) + phi(x)`. **What the paper adds — and what is genuinely new in
this repo — is the risk generalization:** the re-anchored `psi_k`, Algorithm 2
that proxes it, and the `RiskEnvelope` abstraction. We ported the scaffold and
built the new part on top. Part VII gives the line-by-line ledger.

---

## Part II — Setup and your first solve

### II.1 The package has almost no dependencies

`Project.toml`:

```toml
[deps]
LinearAlgebra = ...
Printf = ...
```

That's it. No optimization framework, no autodiff, no plotting. The compat is
`julia = "1.10"`. This is deliberate: the package is a self-contained
reimplementation, and the demo deck generates its figures with the same two
stdlib deps plus `Random`. Keeping the footprint tiny is a feature you should be
prepared to defend (see Part XIV).

### II.2 Run the tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The suite is ~1300 lines across 14 files and (as last recorded) passes ~7985
assertions. Tests are the real specification — when in doubt about how a piece is
meant to be used, open the matching `test/test_*.jl`.

### II.3 The five-ingredient recipe

Every problem is the same assembly. You pick one object for each slot of
`J = f + suppA∘F + phi`, wrap them in a `RiskProblem`, and call `solve`:

```julia
using ROLOpt

f   = ...   # <: SmoothObjective   — the smooth part
F   = ...   # <: VectorMap         — the map into Y
phi = ...   # <: ProxObjective     — the prox-friendly convex part
A   = ...   # <: RiskEnvelope      — defines suppA
# optional X, Y :: HilbertSpace    — default EuclideanSpace

prob = RiskProblem(f, F, phi, A)              # ; X=..., Y=... if non-Euclidean
res  = solve(prob, x0; opts = TROptions{Float64}())

res.x          # solution vector
res.J          # objective value J(x)
res.hk         # stationarity metric h_k (converged when ≤ opts.htol)
res.converged  # Bool
res.iters      # outer iterations
res.status     # :converged | :prox_failure | :stalled | :maxiter
res.counters   # nfval, ngrad, nhess, npsi, nprox, ...
```

### II.4 Your first solve (recovers a closed form)

`½‖x‖² + max(0, Σxᵢ + 0.5)`. The envelope `A = [0,1]` makes `suppA(y) = max(0,y)`,
so the second term is `(Σx + 0.5)₊`. The optimum is `x* = −1/6·𝟙`:

```julia
n   = 3
f   = TikhonovObjective(1.0)            # (τ/2)‖x‖², τ = 1
F   = AffineMap(ones(1, n), [0.5])      # F(x) = Σx + 0.5  (scalar-valued)
phi = ZeroProx{Float64}()               # no phi
A   = IntervalEnvelope(0.0, 1.0)        # suppA(y) = max(0, y)

prob = RiskProblem(f, F, phi, A)
res  = solve(prob, zeros(n); opts = TROptions{Float64}(verbose = 1))
res.x        # ≈ [-1/6, -1/6, -1/6]
```

`verbose = 1` prints the iteration table — watch `hk` drop below `1e-8` and the
radius `delta` grow as steps succeed. This is the README example; the demo's
`risk1` (`demo/generate_evidence.jl:114`) is the same problem.

---

## Part III — The mental model you must hold

Before reading code, internalize **three conventions**. They are inherited from
ROL, they are correct, and most "why is there a `riesz!` here?" confusion
dissolves once you hold them.

### III.1 Primal vs. coordinate (dual) vectors

A function's **gradient is a functional** — it lives in the dual space `X*`.
Numerically it arrives as a *coordinate* vector (partial derivatives). To use it
as a *direction* you live in (to take a step `x − t·g`), you must map it back to
the primal space with the **Riesz map** `M⁻¹`.

The codebase follows ROL's split exactly (`src/objectives.jl:3-9`):

- `gradient!`, `hessVec!`, `applyAdjointJacobian!`, `applyAdjointHessian!`
  return **coordinate / dual** vectors.
- `value!`, `applyJacobian!`, `prox!`, `project!` return **primal** vectors.
- You convert coordinate → primal with `riesz!(out, X, g)` and primal →
  coordinate with `metric!(out, X, v)`.

### III.2 The metric is the inner product, and it is *not* the dot product

`X` and `Y` are Hilbert spaces with a **diagonal metric** `M`:
`(a,b)_X = aᵀ M b`. For a PDE control space `M = diag(element volumes)`; for a
probability space `Y = L²(Ω,P)`, `M = diag(probabilities)`. The Euclidean case
`M = I` (plain `dot`) is just one special case — the `W = Nothing`
specialization that compiles to `dot`/`copyto!` with zero overhead.

**Why this matters and is not pedantry:** the paper's headline result is **mesh
independence** (`main.tex:1614`, `main.tex:1674`) — iteration counts invariant to
discretization size. That only holds when norms and inner products are the
*function-space* ones (mass-matrix weighted), not the Euclidean coefficient norm.
The paper says so explicitly: "our implementation uses infinite-dimensionally
consistent scaled inner products and norms for X and Y" (`main.tex:1675`). The
`HilbertSpace` type is precisely that consistency, made first-class.

### III.3 The adjoint is not the transpose

Under metrics `M_X`, `M_Y`, the operator adjoint of `A` satisfies
`(Av, θ)_Y = (v, A*θ)_X`, which works out to (`src/spaces.jl:17`):

```
A*θ = riesz_X( Aᵀ (M_Y θ) ).
```

The bare matrix transpose `Aᵀ` is the adjoint **only** when both metrics are `I`.
The dual prox (Algorithm 2) and the curvature term both need the true adjoint, so
you'll see `metric!` then `applyAdjointJacobian!` then `riesz!` chained together.

### III.4 The include order is the dependency order

`src/ROLOpt.jl:17-25` includes files bottom-up. Read them in this order — each
builds only on the ones before:

```
spaces.jl        → objectives.jl → riskenvelope.jl → check.jl
   → psimodel.jl → model.jl → subsolvers.jl → inexact.jl → trustregion.jl
```

---

## Part IV — File-by-file walkthrough

Total source is ~2090 lines across 10 files. Here is every one, in include order.

### IV.1 `src/spaces.jl` (68 lines) — the Hilbert space / metric

**What it is.** The `HilbertSpace{T,W}` type and the metric operations everything
else is phrased in terms of. `W = Nothing` is Euclidean; `W <: AbstractVector` is
`M = diag(weights)`.

**Key operations:**

- `inner(X, a, b)` — `(a,b)_X = Σ wᵢ aᵢ bᵢ` (or `dot` when Euclidean).
- `riesz!(out, X, g)` — `M⁻¹ g`, coordinate → primal (`out .= g ./ w`).
- `metric!(out, X, v)` — `M v`, primal → coordinate (`out .= w .* v`).
- `spnorm(X, a)` — `√(a,a)_X`.
- `apply(g, v) = dot(g, v)` — the **metric-free** duality pairing `⟨g,v⟩`; never
  touches `M`.
- `proxweights(X)` — the diagonal of `M` (or `nothing`) for separable prox
  scaling.

**ROL vs new.** The *concept* mirrors ROL's `Vector::dot` / `Vector::dual`
(`src/spaces.jl:6`). The diagonal-metric Julia implementation is new but tiny.

**The one restriction to defend.** The metric is **diagonal only**
(`src/spaces.jl:8-13`). Why: a consistent P1 mass matrix is non-diagonal, which
would make even a separable `phi` prox non-separable and kill the closed-form box
projection. Practitioners use the lumped/diagonal metric for exactly this reason,
and the paper uses element volumes / probabilities — both diagonal.

**Identity to remember** (tested in `test_weighted.jl:26`):
`inner(X, v, riesz!(X, g)) == dot(g, v)` and `riesz ∘ metric = id`.

### IV.2 `src/objectives.jl` (278 lines) — the catalog of `f`, `F`, `phi`

**What it is.** Three abstract types and their concrete instances — the building
blocks you assemble into a problem.

**`SmoothObjective{T}`** — the smooth `f`. Interface: `value`, `gradient!`,
`hessVec!`. Instances:
- `ZeroObjective` — `f ≡ 0`.
- `DiagonalQuadratic(a, b)` — `Σ ½aᵢxᵢ² + bᵢxᵢ` (matches ROL `test_01`'s smooth term).
- `QuadraticObjective(H, b)` — `½xᵀHx + bᵀx`.
- `TikhonovObjective(τ)` — `(τ/2)‖x‖²`.

**`VectorMap{T}`** — the smooth `F : X → Y`. Interface: `value!`,
`applyJacobian!`, `applyAdjointJacobian!`, `applyAdjointHessian!`, `rangedim`.
Instances:
- `AffineMap(A, c)` — `F(x) = Ax + c`, so `F' = A` (constant), `F'' = 0`.
- `QuadraticMap(Q, a, c)` — per output `j`, `Fⱼ(x) = ½xᵀQⱼx + aⱼᵀx + cⱼ`. This is
  the one with nonzero `F''`, used to exercise the curvature term.

**`ProxObjective{T}`** — the prox-friendly `phi`. Interface: `value`, `prox!`,
and the **metric-aware** wrapper `proxX!`. Instances:
- `ZeroProx` — `phi ≡ 0`, prox is identity.
- `L1Objective(w, shift)` — `Σ wᵢ|xᵢ − shiftᵢ|`; prox is soft-threshold.
- `BoundConstraint(lo, hi)` — box indicator; prox is `clamp`.
- `BoxL1Objective(w, lo, hi, shift)` — `L1 + box indicator`; prox is
  soft-threshold-then-clamp, the **proven closed form** (not a generic composition
  of two proxes).

**The metric-aware prox** is the subtle bit (`src/objectives.jl:195-200`). For a
diagonal metric, `prox^X_{t·phi}(z)` is the per-component Euclidean prox with step
`t/wᵢ`. So `proxX!` divides the step vector by the weights and calls the ordinary
`prox!`. The box-projection part of `phi` is metric-independent (clamping doesn't
care about the inner product); only the L1 / quadratic part scales.

**ROL vs new.** The prox *formulas* are ports: `L1Objective` ports ROL
`l1Objective` (`:213`), `BoundConstraint` ports `Bounds` (`:237`), `BoxL1Objective`
is the paper's sparse-control `phi` (`:254`). The abstract-type structure and the
metric-aware `proxX!` wrapper are new (Julia + the weighted-space generalization).

### IV.3 `src/riskenvelope.jl` (198 lines) — the net-new heart

**What it is.** The `RiskEnvelope{T}` abstraction — the genuinely novel content
that has **no ROL `TypeP` equivalent**, because it is what the paper invents.

Interface (all phrased with respect to the `Y` metric): `suppA(A, Y, y)`,
`project!(out, A, Y, y)` (metric projection onto `A`), `bound(A, Y)` (`= MA =
sup_{θ∈A} ‖θ‖_Y`), `initdual!(theta, A)` (a feasible starting dual).

**`IntervalEnvelope(a, b, n)`** — the box `A = [a,b]ⁿ`. Then
`suppA(y) = Σ wᵢ·max(a·yᵢ, b·yᵢ)`, `projA = clamp(y, a, b)`. For the paper's
sparse-control example `A = [0,1]`, scalar, this is `suppA(y) = max(0, y)` — the
positive part.

**`ConvexCombinationEnvelope(λ, p, probs)`** — the **E + CVaR** envelope, the real
prize. It represents `R(y) = (1−λ)E[y] + λ·AVaR_p[y]`, whose envelope is
`A = {θ : Σ pᵢθᵢ = 1, l ≤ θᵢ ≤ u}` with `l = 1−λ`, `u = (1−λ) + λ/(1−p)`.

Two non-trivial algorithms live here:
- **`suppA` by water-filling** (`:109`). The support function is an LP over a box
  intersected with a budget hyperplane — a continuous knapsack. Start at `θᵢ = l`,
  then raise `θᵢ` toward `u` in *descending* order of value density `dᵢ = wᵢyᵢ/pᵢ`
  until the budget `Σ pᵢθᵢ = 1` is met. (Note: this evaluation allocates a sort —
  a bounded `O(n log n)`, the one deliberate non-zero-alloc path, documented at
  `:114-122`.)
- **`projA` by KKT bisection** (`:160`). Project onto `A` in the `Y` metric via the
  shift `θᵢ = clamp(yᵢ + ν·pᵢ/wᵢ, l, u)`, with `ν` the root of the budget residual
  `Σ pᵢθᵢ = 1`, found by a closure-free, allocation-free bisection (it runs in the
  hot dual loop).

**Why metric-generic matters here.** `suppA(A, Yprob, y)` equals `R(y)` *exactly*
when `Y = WeightedSpace(probs)`; the same envelope under a Euclidean `Y` computes
a probability-weighted variant. The demo checks both against an independent
water-filling reference (`demo/generate_evidence.jl:188`).

### IV.4 `src/check.jl` (112 lines) — finite-difference derivative checks

**What it is.** Verification tools, ports of ROL's `checkGradient` /
`checkHessVec` / `checkHessSym` / `checkApplyJacobian` / `checkAdjointConsistency`.
Each returns the **minimum error over a ladder of step sizes** so a test can
assert it's small.

- `check_gradient(f, x, v)` — `⟨∇f(x), v⟩` vs. central FD of `t ↦ f(x+tv)`.
- `check_hessvec(f, x, v)` — `∇²f(x)v` vs. central FD of `∇f`.
- `check_jacobian(F, x, v)` — `F'(x)v` vs. central FD of `F`.
- `check_adjoint(F, x, v, θ)` — the identity `⟨F'v, θ⟩ = ⟨v, F'*θ⟩` (metric-free).
- `check_hess_sym(f, x, u, v)` — `⟨Hu, v⟩ = ⟨Hv, u⟩`.

**Generic-precision detail** (`:11`): the step sizes are converted to `eltype(x)`,
so under BigFloat / Float32 the differences run in the working precision instead
of silently promoting through Float64. Central differences are exact (to roundoff)
on the quadratic test objectives — which is why the demo's identity table shows
machine-epsilon errors.

### IV.5 `src/psimodel.jl` (210 lines) — **Algorithm 2** (the inner solver)

**What it is.** The re-anchorable nonsmooth model term `psi_k` and its proximity
operator, computed by a nonmonotone spectral projected-gradient method on the
dual. This is the paper's net-new algorithmic contribution.

**The model term** (`:126`):
```
psi_k(x) = suppA( ℓ_k(x) ) + phi(x),   ℓ_k(x) = A_k(x − x_k) + b_k.
```
`reanchor!(psi, x_k)` sets `x_k` and `b_k = F(x_k)` once per outer iteration
(`:119`).

**The dual derivation** (paper `main.tex:657`, eq. `dualprox`). The prox of
`psi_k` at argument `z`, step `t = r`, is found by *maximizing* a concave dual
over `θ ∈ A`:
```
p(θ)      = prox_{t·phi}( z − t·A_k*θ )                  (the primal recovery)
∇d(θ)     = A_k( p(θ) − x_k ) + b_k = ℓ_k(p(θ))          (the dual gradient, in Y)
d(θ)      = (1/2t)‖p(θ) − z‖²_X + phi(p(θ)) + (θ, ℓ_k(p))_Y
```
At the optimal `θ̄`, `prox_{t·psi_k}(z) = p(θ̄)`. So the inner solver climbs `d` and
reads off the primal.

**The inner kernel `_dual_d!`** (`:137`) computes `d(θ)`, writing `p(θ)` and
`∇d(θ)` into buffers. Watch the metric dance, which is exactly Part III.3:
`metric!` (form `M_Y θ`) → `applyAdjointJacobian!` (`Aᵀ(M_Y θ)`) → `riesz!`
(`A_k*θ`, now primal) → `proxX!` (metric-aware prox of `phi`).

**`prox!`** (`:155`) is **Algorithm 2** itself — a Barzilai–Borwein spectral
projected gradient with a nonmonotone, safeguarded quadratic-interpolation line
search:
- Projected ascent step `s = projA(θ + γ·∇d) − θ`.
- BB step length `γ = λ_prev‖s_prev‖² / (∇d_prev − ∇d, s_prev)`.
- Nonmonotone Armijo line search against the best of the last `M` dual values
  (the `_dq_*` fixed-capacity ring at `:69-87` keeps that memory allocation-free).
- Stops when the projected-gradient norm `‖s‖_Y/γ ≤ tol`.
- Returns an `isbits` `DualSPGResult{T}` (iters, final pgnorm, converged) — so the
  call allocates nothing. The optimal dual is left in `psi.theta` as a warm-start
  cache (snapshot with `copy` if you need it to survive the next `prox!`).

**ROL vs new.** The SPG line-search *template* is ported from ROL's
`SpectralGradientAlgorithm` (`:21`). The *problem it is applied to* — the dual of
the `psi_k` prox over the risk envelope — is the paper's Algorithm 2. This file is
the cleanest example of "ROL skeleton, paper flesh."

### IV.6 `src/model.jl` (89 lines) — the smooth quadratic model `q_k`

**What it is.** The smooth part of the local model,
`q_k(x) = ½(B_k(x−x_k), x−x_k)_X + (g_k, x−x_k)_X`, and the rule for the model
Hessian `B_k`.

Two choices of `B_k`:
- **Default** (`F = nothing`): `B_k = ∇²f(x_k)` (the smooth Hessian).
- **Curvature on** (`use_curvature = true`): `B_k = ∇²f(x_k) + (θ_k, F''(x_k)·)_Y`,
  the `x`-Hessian of the Lagrangian `(θ_k, F(x))_Y`.

Either is a legitimate bounded model Hessian — the convergence theory permits any
bounded `B_k`. Key invariant (`:13-22`): when curvature is on, `θ_k` is **frozen**
from the once-per-iteration stationarity prox (`freeze_curvature!`) and held
constant for the whole outer iteration, even though the many proxes during the
subproblem solve mutate `psi.theta`. The stored multiplier is the *metric* dual
`M_Y·θ_k` (so the weighted-`Y` metric isn't dropped — `:71-75`).

`setmodel!(m, xk, gcoord)` re-anchors and Riesz-maps the gradient; `modelhessvec!`
applies `B_k v` as a coordinate vector.

**ROL vs new.** The plain quadratic TR model is standard. The `(θ_k, F'')`
curvature term is from the paper's Lagrangian model (`main.tex:475-493`).

### IV.7 `src/subsolvers.jl` (504 lines) — the trial-step machinery

**What it is.** The biggest file, and the most ROL-shaped. It computes the trial
step that satisfies the trust-region conditions, plus the stationarity metric and
the evaluation counters.

**`Counters`** (`:11`) — `nfval`, `ngrad`, `nhess`, `npsi`, `nprox`, `nspg`,
`nproxfail`. **These columns deliberately mirror the paper's result tables**
(`:10`) — compare to Table 3 (`main.tex:1650`), whose columns are
`iter nfval ngrad nhess npsi nprox aprox`. We built the ruler to match the paper.

**`pgstep!`** (`:50`) — one proximal-gradient step
`prox_{t·nobj}(x − t·dg)`. `nobj` is a `psi_k` here, so each call runs Algorithm 2.

**`hstat!`** (`:64`) — the **stationarity metric**
`h_k = ‖prox_{t0·psi_k}(x − t0·dg) − x‖_X / t0` (paper eq. `hk`, `main.tex:524`).
This is the convergence yardstick: `h_k → 0` means stationary. It returns an `ok`
flag — if the inner prox didn't converge, `h_k` is untrustworthy and the solver
must not certify convergence on it.

**`dcauchy!`** (`:111`) — the **Cauchy point**, the benchmark trial step (paper
`main.tex:620`). A bidirectional proximal search on the step length `α`:
interpolate (shrink `α`) until the model decrease is sufficient, or extrapolate
(grow `α`) while it keeps helping. Port of ROL `dcauchy`, but generalized so the
prox inside is `psi_k`'s iterative dual solve, not a closed form. It returns
`worst_pgnorm` — the worst dual-SPG residual among its proxes — so the caller can
tell a "the inner prox barely missed tolerance" step from a "the inner prox is
garbage" step.

**`dspg!`** (`:201`) — the **spectral proximal-gradient improver**
(`subsolver = :spg`). Starting from the Cauchy point, take BB proximal-gradient
steps to reduce the model further, capping each step at the TR boundary
analytically (no Brent root-find). The model only decreases, so the
fraction-of-Cauchy-decrease guarantee survives. Port of ROL `dspg2`.

**`dncg!` + `dbls!`** (`:421`, `:291`) — the **nonlinear-CG improver**
(`subsolver = :ncg`). `dncg!` takes prox-gradient / CG steps; each step length
comes from `dbls!`, a 1-D Brent line search of the TR model along the search
direction. Seven β-formulas are available (`_ncg_beta`, `:387`); default is
Dai–Yuan+. Ports of ROL `dncg` / `dbls`.

> A subtle bug fixed during the build (commit `f6f0cdb`, memorialized at
> `:299-302`): when the Cauchy point already sits *on* the TR boundary, the
> feasible interval for `dbls!` is empty (`tmax ≤ 0`), and the quadratic-upper-bound
> minimizer becomes a `0/0` NaN. The fix is the early return at `:302`.

**ROL vs new.** This file is ~all ported control flow (`dcauchy`/`dspg2`/`dncg`/
`dbls`/`pgstep`), retyped into Julia and **generalized from `phi` to `psi_k`**.
The `worst_pgnorm` plumbing and the graded prox-reliability policy are new — they
exist because, unlike ROL's closed-form `phi` prox, our `psi_k` prox can fail.

### IV.8 `src/inexact.jl` (168 lines) — Conditions 1 & 2

**What it is.** The machinery that lets the solver use *inexact* values and
derivatives of `f` and `F` — the feature that makes the method viable for
expensive (PDE) `F`. The paper's two conditions (`main.tex:551`, `:565`):

```
Condition 1 (model data accuracy at x_k):
  ‖g_k − ∇f(x_k)‖_X        ≤ κ_grad · min(h_k, Δ_k)
  ‖b_k − F(x_k)‖_Y         ≤ κ_val  · min(h_k, Δ_k²)      ← note Δ²
  ‖A_k − F'(x_k)‖_{L(X,Y)} ≤ κ_jac  · min(h_k, Δ_k)
Condition 2 (computed reduction):
  |ared − cred|            ≤ κ_obj · [η · min(pred, ζ_k)]^ζ,   ζ_k → 0.
```

**Design philosophy.** `prob.F` is *never* replaced — the exact `F` stays the
ground truth for the objective. The inexactness is injected through an
`InexactMap` (`:33`) that wraps the exact `F` and adds a *controlled, recorded*
perturbation: `A_k = F'(x_k) + E`, `b_k = F(x_k) + e_b`. The `InexactOracle`
(`:84`) sets the perturbation norms to a fraction `ρ` of each tolerance and
**records the realized errors**, so the contract is checked directly, not inferred
from noise. The audit trail is `oracle.records :: Vector{InexactRecord}` (`:71`),
which the demo turns into the slide-5 table.

`opnorm_metric` (`:21`) computes the true operator norm `‖E‖_{L(X,Y)}` under the
diagonal metrics (the spectral norm of `diag(√wY)·E·diag(1/√wX)`), so the recorded
Jacobian error is the metric one, not a naive matrix norm.

**ROL vs new.** This is mostly new wiring. The *conditions* are the paper's; the
auditable oracle/map design is this repo's.

### IV.9 `src/trustregion.jl` (404 lines) — **Algorithm 1** (the driver)

**What it is.** The outer loop — `solve`, `RiskProblem`, `TROptions`, `TRResult` —
the thing a user actually calls. Blueprint: ROL
`TypeP_TrustRegionAlgorithm_Def.hpp::run`.

**`RiskProblem`** (`:17`) bundles `(f, F, phi, A, X, Y)`. **`TROptions{T}`**
(`:75`) holds every knob; the defaults follow the paper's Numerics section
(`main.tex:1472`): `Δ₁ = 10`, `η₁ = 1e-4`, `η₂ = 0.5`, `γ₁ = γ₂ = 0.25`,
`γ₃ = 10`, stop at `h_k ≤ 1e-8`. `subsolver ∈ {:cauchy, :spg, :ncg}`;
`use_curvature::Bool`; nested option structs for the dual SPG, SPG, NCG.

**The ratio test** `analyze_ratio_red` (`:50`) — port of ROL `TRUtils::analyzeRatio`.
It adds a roundoff floor `ε = epsi·max(1,|fold|)` to both reductions so that near
the solution, when both are at the roundoff floor, it declares `ρ = 1` (success)
instead of producing garbage from catastrophic cancellation. Returns a flag
(`TR_SUCCESS`, `TR_POSPREDNEG`, etc.) the acceptance logic reads.

**The function-barrier design** (`:364` `solve` → `:260` `_solve_exact!` /
`:312` `_solve_inexact!`). `solve` constructs the concrete `psi`/`model`/`imap`
types, then dispatches *once* into a typed kernel. This is the key
type-stability move: the hot loop never dynamically dispatches on a union-typed
model. If you remember one Julia-engineering decision in this repo, remember this
one (Part XIV will ask you about it).

**The exact kernel `_solve_exact!`** (`:260`) is Algorithm 1 in the flesh:
1. Refresh the model at `x` (`_refresh_exact!`, `:164`): gradient → `setmodel!` →
   `reanchor!` → `hstat!` → `freeze_curvature!`. Get `h_k`.
2. Loop: if `h_k ≤ htol` and the prox converged → done. If the radius collapsed
   below iterate precision → stalled. Else compute a trial step
   (`_trial_step!`, `:214`), form the true `ared = J(x) − J(x⁺)`, run the ratio
   test, accept/reject (`_accept_decision`, `:241`), update `Δ`.
3. On accept, refresh the model at the new point.

**The inexact kernel `_solve_inexact!`** (`:312`) is the same skeleton with two
differences: (a) the model data is rebuilt by a warm-started fixed point
(`_refresh_inexact!`, `:183`) whose Condition-1 tolerances depend on `(h_k, Δ)`,
followed by a *postcondition* injection so the data is consistent with the
*reported* `h_k`; and (b) the accept/reject test uses the **computed** reduction
`cred = ared + bounded error` instead of `ared`, with `ζ_k → 0` advancing every
outer iteration (`:350`).

**The auto initial radius** (`:274-278`) is a small but important fix (commit
`b03484a`): when `Δ₁ ≤ 0`, the radius is seeded from
`max(‖g‖_X, ‖prox-grad step‖_X)` — using the *nonsmooth* stationarity step, not
just the smooth gradient, so a problem with `∇f = 0` but an active risk/`phi` term
doesn't false-stall on iteration 1.

**`TRResult.status`** (`:114-121`) is honest about *why* it stopped:
`:converged` (h_k certified), `:prox_failure` (the inner prox capped out, so h_k
is unreliable), `:stalled` (radius collapsed before reaching htol), or `:maxiter`.

---

## Part V — One `solve` call, traced end to end

Putting it together, here is the exact call graph of `solve(prob, x0; opts)` in
the **exact, `:spg`** configuration:

```
solve                                     (trustregion.jl:364)
├─ validate opts / oracle
├─ build psi   = PsiModel(phi, A, F, X, Y) (psimodel.jl)
├─ build model = QuadraticModel(f, X)      (model.jl)
└─ _solve_exact!(prob, opts, psi, model, bw, cnt)        (trustregion.jl:260)
   ├─ _refresh_exact!                                    (:164)
   │  ├─ gradient!(f) ; riesz!                           → primal g_k
   │  ├─ setmodel!(model, x, gcoord)                     (model.jl:57)
   │  ├─ reanchor!(psi, x)  → b_k = F(x)                 (psimodel.jl:119)
   │  ├─ hstat!  → h_k, ok                               (subsolvers.jl:64)
   │  │  └─ pgstep! → prox!(psi, ...)  ===  ALGORITHM 2  (psimodel.jl:155)
   │  └─ freeze_curvature!(model, psi.theta)             (model.jl:64)
   └─ while not converged / stalled / maxit:
      ├─ _trial_step!                                    (:214)
      │  ├─ dcauchy!  → Cauchy step (many prox! = Alg 2) (subsolvers.jl:111)
      │  ├─ dspg!     → improve it  (more prox! = Alg 2) (subsolvers.jl:201)
      │  └─ objective_value!  → true J(x⁺) using exact F (:138)
      ├─ analyze_ratio_red(ared, pRed, ...)              (:50)
      ├─ _accept_decision(rho, flag, worstpg, opts)      (:241)
      └─ accept ? (copy x⁺, grow Δ, _refresh_exact!) : (shrink Δ)
```

Two things to notice. First, **Algorithm 2 (`prox!`) runs many times per outer
iteration** — once for `h_k`, then repeatedly inside `dcauchy!` and `dspg!`. That
is why `nprox` and the average inner-iteration count `aprox` are headline metrics
in the paper (Table 3). Second, the **true objective always uses `prob.F`**
(`objective_value!`, `:138`), never the linearization `ℓ_k` — the model and the
truth are kept strictly separate (tested at `test_risk_finitedim_p1.jl:65`).

---

## Part VI — The math-to-code map

| Paper object | Paper ref | Code |
|---|---|---|
| Problem `min f + suppA∘F + phi` | `main.tex:151` | `RiskProblem` (`trustregion.jl:17`) |
| Risk reformulation `R(F(x))` | `main.tex:164` | `RiskEnvelope` + `suppA` (`riskenvelope.jl`) |
| Stationarity condition | `main.tex:323` | — (characterized by `h_k=0`) |
| `psi_x` (linearized nonsmooth) | `main.tex:334` | `PsiModel` (`psimodel.jl`) |
| Prox-difference lemma | `main.tex:387` | (theory behind convergence) |
| Lagrangian model `L_k`, `q_k`, `ℓ_k` | `main.tex:483-490` | `QuadraticModel` (`model.jl`) + `psi` |
| Local model `m_k = q_k + psi_k` | `main.tex:496` | `q_k` (`model.jl`) + `psi_k` (`psimodel.jl`) |
| Trial-step conditions (rad, fcd) | `main.tex:516-520` | `_trial_step!` / `_accept_decision` |
| Stationarity metric `h_k` | `main.tex:524` | `hstat!` (`subsolvers.jl:64`) |
| Ratio `ρ_k = ared/pred` | `main.tex:534` | `analyze_ratio_red` (`trustregion.jl:50`) |
| Condition 1 (inexact grad/val/jac) | `main.tex:551` | `set_tols!`, `inject_*` (`inexact.jl`) |
| Condition 2 (computed reduction) | `main.tex:565` | `computed_reduction` (`inexact.jl:155`) |
| **Algorithm 1 (TR-Risk)** | `main.tex:591` | `_solve_exact!` / `_solve_inexact!` |
| Cauchy point `x_k^c` | `main.tex:631` | `dcauchy!` (`subsolvers.jl:111`) |
| Dual prox problem | `main.tex:657` | `_dual_d!` (`psimodel.jl:137`) |
| **Algorithm 2 (ψ_k prox)** | `main.tex:687` | `prox!` (`psimodel.jl:155`) |
| Trial improvers (CG / SPG) | `main.tex:774` | `dncg!`, `dspg!` (`subsolvers.jl`) |
| Global convergence `liminf h_k = 0` | `main.tex:913` | (guarantees the `htol` stop terminates) |
| Numerics defaults (`Δ₁`, `η`, `γ`, htol) | `main.tex:1472` | `TROptions` defaults (`trustregion.jl:98`) |
| E + CVaR envelope | `main.tex:1505` | `ConvexCombinationEnvelope` |
| Sparse-control example (`A=[0,1]`, BoxL1) | `main.tex:1618` | `test_risk_finitedim_p1.jl` (finite-dim analogue) |
| Table-3 columns (iter…nprox aprox) | `main.tex:1650` | `Counters` (`subsolvers.jl:11`) |

---

## Part VII — Provenance ledger: ROL vs new

**Nothing is copy-pasted** — ROL is C++, this is Julia, so every line was retyped
by hand, restructured for multiple dispatch, made type-stable / zero-alloc /
generic over `T`. The honest axis is "faithful transliteration of existing ROL
logic" vs. "genuinely new design." Roughly, by content:

**Layer 1 — ROL `TypeP` logic, retyped and generalized (~45–50% of lines).**
- `subsolvers.jl` — `dcauchy`/`dspg2`/`dncg`/`dbls`/`pgstep`.
- `trustregion.jl` — the `run` loop skeleton; `analyzeRatio`.
- `check.jl` — the FD check routines.
- The prox closed forms in `objectives.jl` (`l1Objective`, `Bounds`, `boxL1`).

Even here it isn't mechanical: every ported routine was **generalized from `phi`
to `psi_k`**, so its inner prox is now an *iterative dual algorithm that can fail*
— which is why `worst_pgnorm`, the graded reliability policy, and the
`:prox_failure` status exist (none in ROL).

**Layer 2 — the paper's net-new algorithm, no ROL equivalent (~28%).**
- `riskenvelope.jl` — entirely new (envelopes, water-filling, KKT projection).
- `psimodel.jl` — Algorithm 2 applied to the dual risk prox (ROL SPG template,
  paper's problem).
- `inexact.jl` — the auditable Conditions 1 & 2 realization.
- the `(θ_k, F'')` curvature term in `model.jl`.

**Layer 3 — Julia engineering with no C++ analog (~22%).**
- the `HilbertSpace` diagonal-metric abstraction;
- the function-barrier `_solve_exact!`/`_solve_inexact!` split;
- the `_SolveBuffers` / `PsiWorkspace` zero-alloc scratch and the `isbits`
  `DualSPGResult`;
- generic `T` (BigFloat / Float32);
- the `_refresh_inexact!` warm-started fixed point (new logic, not a port).

**One-liner:** about half the lines are recognizably ROL `TypeP` control flow
retyped into Julia and generalized; the other half — the risk envelopes,
Algorithm 2's dual prox, the inexactness wiring, the metric generalization — has
no ROL counterpart because the paper is what invented it.

> Caveat to state honestly: this split is an informed estimate from the in-file
> `Blueprint:`/`Port of` citations plus build knowledge, **not** a measured
> line-diff against Trilinos `46b83e0` (which isn't checked out here). The rigorous
> version would clone ROL at that commit and map routine by routine.

---

## Part VIII — The three advanced axes

The solver is one algorithm, but it generalizes along three independent axes.
Understanding each as "off by default, switched on by one knob" is the cleanest
way to hold the whole thing.

### VIII.1 The metric axis — Euclidean → weighted

Default `X`, `Y` are `EuclideanSpace`. Pass `WeightedSpace(w)` to use a diagonal
metric:

```julia
Yp   = WeightedSpace(probs)                      # Y = L²(Ω,P)
prob = RiskProblem(f, F, phi, A; Y = Yp)         # weighted Y, Euclidean X
```

This is the mesh-independence axis (Part III.2). Tested adversarially in
`test_weighted.jl` — those tests are *designed to fail* if any operation silently
assumes `M = I`.

### VIII.2 The inexactness axis — exact → inexact oracle

Default is exact. Pass an `InexactOracle` to run the Conditions-1&2 variant:

```julia
oracle = InexactOracle(f; kgrad=1.0, kval=1.0, kjac=1.0, kobj=1e2, rho=0.5)
res = solve(prob, x0; opts=TROptions{Float64}(delta0=10.0, subsolver=:spg), oracle=oracle)
oracle.records      # per-iteration audit: realized error vs. each bound
```

`solve` validates `0 < η < min(η₁, 1−η₂)` (Condition 2's requirement,
`trustregion.jl:372`) and that `Δ₁ > 0` (inexact tolerances scale with `Δ`).

### VIII.3 The precision axis — Float64 → Float32 / BigFloat

The whole package is generic over `T <: Real`. Everything threads `T`: the
options (`TROptions{T}`), the FD steps (converted to `eltype(x)`), the dual SPG
tolerances. The demo's "precision ladder" (`generate_evidence.jl:164`) solves the
same LASSO at Float32 / Float64 / BigFloat under precision-scaled tolerances and
shows `‖x − x*‖` tightening with the requested precision.

```julia
solve(prob, zeros(BigFloat, n); opts = TROptions{BigFloat}(htol = BigFloat(1e-20)))
```

### VIII.4 The F-representation axis — explicit / matrix-free / cached-Jacobian

`F` is reached only through the `VectorMap` operator-action interface (`value!`,
`applyJacobian!`, `applyAdjointJacobian!`, optional `applyAdjointHessian!`,
`rangedim`), plus a once-per-anchor `prepare!(F, x_k)` hook (`reanchor!`, before
`value!`). **You choose how `F` is stored by the type you pass; ROLOpt never
infers or assembles `A_k`.** Three representations (`src/usermaps.jl`):

- **`AffineMap` / `QuadraticMap`** — explicit stored Jacobian. The default.
- **`OperatorMap`** — pure matrix-free: you supply `value`/`jvp`/`vjp` (and optional
  `ahp`/`prepare`) callbacks; no `A_k` is stored. For PDE-scale `F` whose Jacobian
  cannot be formed (each apply is a tangent/adjoint solve).
- **`CachedJacobianMap`** — nonlinear `F` whose Jacobian *is* storable: you supply
  `assemble!(Ak, x)`, `prepare!` builds `A_k = F'(x_k)` once per anchor, and applies
  are cached gemvs. Distinct from `AffineMap` (whose `A` is fixed).

```julia
F = OperatorMap{Float64}(; m = dimY,
        value = (b, x)  -> forward_solve!(b, x),      # b = F(x)        (primal Y)
        jvp   = (o, v, x) -> tangent_solve!(o, v, x), # o = F'(x) v      (primal Y)
        vjp   = (o, θ, x) -> adjoint_solve!(o, θ, x), # o = F'(x)ᵀ θ     (coord X, BARE transpose)
        prepare = x -> factorize_state!(x))           # cache the frozen factorization
```

**The #1 pitfall:** `vjp` must return the *bare coordinate transpose* `Aᵀθ` — **no
metric**. The framework forms the true metric adjoint `riesz_X(Aᵀ(M_Y θ))` itself.
Validate a JVP/VJP pair with `check_adjoint` / `check_jacobian` (both run `prepare!`
first), or set `TROptions(validate_F = true)` to run a deterministic adjoint check
once at the start of `solve`. **Cache contract:** `prepare!` owns the linearization
cache; `value!` (also called at *trial* points by the ratio test) must keep its own
separate state and never mutate it, so a rejected step still reuses the `x_k`
Jacobian. **Factorization reuse:** because `x_k` is frozen between `prepare!` calls,
LU-factor the state operator once in `prepare!` and serve every JVP/VJP as a
back-substitution (the VJP reuses the same factors via a transpose solve). Tested in
`test_matfree.jl` (three representations agree on the same map; weighted metric;
wrong-VJP detection; `prepare!`-once-per-anchor; the cache contract under
rejections; a factorization-cache pattern; curvature; Float32/BigFloat).

---

## Part IX — The test suite, file by file

The tests are the executable specification. Each proves something specific:

| File | Proves |
|---|---|
| `test_check.jl` | The FD checks themselves are correct (machine-exact on quadratics). |
| `test_riskenvelope.jl` | `suppA` / `projA` for both envelopes vs. independent references (water-filling, KKT). |
| `test_psimodel_prox.jl` | Algorithm 2 returns the true `prox_{t·psi_k}` (vs. a brute-force optimum). |
| `test_cauchy_decrease.jl` | The Cauchy point satisfies the fraction-of-Cauchy-decrease condition. |
| `test_risk_finitedim_p1.jl` | Full Cauchy-only solve on the finite-dim sparse-control analogue; `h_k ≤ 1e-8`; F'' never called in Phase 1; true objective uses exact F. |
| `test_lasso_reduce.jl` | Recovers ROL `test_01`'s closed-form LASSO optimum. |
| `test_weighted.jl` | Weighted-metric consistency — adversarial: fails if any op assumes Euclidean. |
| `test_improver.jl` | The SPG improver drives a hard CVaR problem below `1e-8` where Cauchy stalls. |
| `test_ncg.jl` | The NCG improver does the same; the β-formulas behave. |
| `test_inexact.jl` | Conditions 1 & 2 hold (realized error ≤ bound) and the inexact solve matches the exact minimizer. |
| `test_curvature.jl` | The `(θ_k, F'')` term is correct and frozen across the subproblem solve. |
| `test_precision.jl` | Float32 / Float64 / BigFloat solves converge to the closed form. |
| `test_perf.jl` | Hot kernels are zero-alloc (`@allocated` guards) and type-stable (`@inferred`). |
| `test_solver_robustness.jl` | Edge cases: empty intervals, collapsed radii, prox failures don't crash or false-certify. |

The clearest one to read first as a "user manual with assertions" is
`test_risk_finitedim_p1.jl` — it sets up a complete problem and checks the four
things that matter.

---

## Part X — The demo evidence pipeline

`demo/` is a 5-slide Beamer + pgfplots deck arguing the implementation is faithful.
The pipeline is **data → LaTeX, no plotting dependency**:

- `demo/generate_evidence.jl` runs problem instances **copied verbatim from the
  passing tests** and emits `demo/data/*.dat` and `demo/tables/*.tex` using only
  `ROLOpt + LinearAlgebra + Random + Printf`.
- `pgfplots` draws the figures from the `.dat`; the `.tex` fragments are dropped
  into `slides.tex`.
- `make` regenerates data + tables and builds `slides.pdf`; `make testcount` runs
  the suite and records the passing count as a committed "last verified" figure so
  the default `make` never needs to run tests.

The five sections of `generate_evidence.jl` map to the five slides:
`section_taylor` (FD/identity checks), `section_closedform` + `section_ladder`
(closed-form optima + precision ladder), `section_cvar` (E+CVaR support function
vs. reference, `h_k` convergence per subsolver), `section_inexact` (Conditions 1&2
audit + match to the exact minimizer).

---

## Part XI — Convergence theory you should be able to recite

You don't need the proofs, but to defend the design you should know the *shape*
of the guarantees (paper §4, `main.tex:786`).

**Two classes of result:**

1. **Global convergence** (`main.tex:808`). Under Assumption 1 (the data
   regularity, `main.tex:275`) and a mild summability condition on `‖B_k‖`,
   Theorem `T:conv` (`main.tex:913`) gives `liminf_{k→∞} h_k = 0`. With a uniform
   bound on `‖B_k‖`, Theorem `T:conv-lim` (`main.tex:995`) upgrades that to a full
   limit `lim h_k = 0`. **Consequence you rely on:** the stopping test `h_k ≤ 1e-8`
   is guaranteed to trigger after finitely many iterations (`main.tex:1478`).

2. **Convergence to stationary points** (`main.tex:1158`). Under extra regularity
   (Mosco convergence of the linearized models), the iterates `{x_k}` accumulate
   at a genuine stationary point of `optprob`, in both strong and weak forms.

**The engine of the proofs** is Lemma `L:proxDiff` (`main.tex:387`) — it bounds how
much the prox of the *approximate* `psi_k` can differ from the prox of the *exact*
`psi_x̄`, in terms of the Condition-1 errors. That is exactly why Condition 1 ties
its tolerances to `min(h_k, Δ_k)`: as you approach stationarity, the allowed
inexactness shrinks at just the rate the proof needs. The key new technical step
versus Baraldi is Lemma `L:success` (`main.tex:815`), which re-derives the
"very successful step" bound for the more general `psi_k` (the support-function
subadditivity argument at `main.tex:876` is the new ingredient).

What this buys the code: the `htol` stop is principled, not heuristic; and the
inexact tolerance schedule in `set_tols!` (`inexact.jl:115`) is the literal
realization of the condition the proof requires.

---

## Part XII — Honest scope: the paper's numerics vs what we built

This must be stated plainly, because it is the most likely place to overclaim.

**The paper's experiments are PDE-constrained** (`main.tex:1445`):

1. **Risk-averse Burgers** (`main.tex:1481`) — a stochastic Burgers equation,
   257-interval P1 FEM, Newton-with-linesearch state solves, SAA with 10,000
   samples, `R = 0.25·E + 0.75·AVaR_0.9`. Table 1 (`main.tex:1522`) shows
   Algorithm 1 beating PD-Risk and a bundle method on PDE-solve count; Table 2
   (`main.tex:1581`) shows the inexact variant relaxing solver tolerances.
2. **Sparse semilinear elliptic control** (`main.tex:1613`) — deterministic,
   `A = [0,1]`, `phi = L1 + box`. Table 3 (`main.tex:1646`) is the **mesh
   independence** study: iteration / eval counts essentially constant from
   60×20 up to 1920×640.

**What this repo does NOT do:** it does not implement either PDE example, does not
reproduce any of the three tables, and does not run ROL C++ side by side. There is
no FEM / PDE stack here and no buildable ROL C++ to diff against. The repo is a
**finite-dimensional proof of concept.**

**What it DOES validate** (none of which is the paper's experiment): closed-form
optima we can derive ourselves (the `−1/6` kink, ROL `test_01`'s LASSO formula);
the E+CVaR support function against an independent water-filling reference; and the
inexactness *contract* (Conditions 1 & 2 realized-error ≤ bound) on finite-dim
instances. So "matches the paper" means **theory + structure + analytic optima**,
not "reproduced Table 1."

**The honest bridge to the paper.** Note that example 2 — `A = [0,1]`, scalar
`F(z) = w − (1/|D_o|)∫ S(z)`, `phi = L1 + box` — is *structurally identical* to our
`test_risk_finitedim_p1.jl` instance (`IntervalEnvelope(0,1)`, scalar
`QuadraticMap`, `BoxL1Objective`). Ours is the finite-dimensional shadow of the
paper's §5.2: same objective shape, same envelope, same `phi`, minus the PDE that
produces `F`. And our `Counters` are Table 3's columns. The cleanest path to a
*real* comparison would be to implement that semilinear elliptic `F` with a FEM
discretization (Gridap-friendly), wire its adjoint into the `VectorMap` interface,
and run the mesh-refinement sweep to check the iteration count stays flat.

---

## Part XIII — Glossary: symbol ↔ code name

| Paper symbol | Meaning | Code |
|---|---|---|
| `J` | full objective | `objective_value!` |
| `f` | smooth term | `prob.f :: SmoothObjective` |
| `F` | smooth map `X→Y` | `prob.F :: VectorMap` |
| `phi` | prox-friendly convex term | `prob.phi :: ProxObjective` |
| `A` (`𝔄`) | risk envelope | `prob.A :: RiskEnvelope` |
| `σ_A` | support function `suppA` | `suppA(A, Y, y)` |
| `R` | coherent risk measure | (envelope `A` encodes it) |
| `ψ_x̄`, `ψ_k` | (re-anchored) nonsmooth model | `PsiModel`, `value(psi, x)` |
| `ℓ_k` | linearized map `A_k(x−x_k)+b_k` | inside `_dual_d!` / `value(psi,·)` |
| `q_k` | quadratic smooth model | `QuadraticModel` |
| `m_k` | local model `q_k + ψ_k` | (sum in `_trial_step!`) |
| `B_k` | model Hessian | `modelhessvec!` |
| `g_k` | (approx) gradient | `model.gk` (primal), `gcoord` (coord) |
| `A_k`, `b_k` | (approx) `F'(x_k)`, `F(x_k)` | `psi.F` / `imap.E`, `psi.bk` |
| `θ`, `θ_k` | dual / frozen dual | `psi.theta`, `model.theta` |
| `h_k` | stationarity metric | `res.hk`, `hstat!` |
| `Δ_k` | trust-region radius | `delta` in the kernels |
| `ρ_k` | reduction ratio | `rho` in the kernels |
| `t_k`, `r` | prox / Cauchy step param | `opts.t0`, `t` |
| `ared`, `pred`, `cred` | actual / predicted / computed reduction | `ared`, `pRed`, `cred` |
| `κ_grad`, `κ_val`, `κ_jac`, `κ_obj` | Condition constants | `oracle.kgrad/kval/kjac/kobj` |
| `M_𝔄` | envelope bound | `bound(A, Y)` |
| `iter, nfval, ngrad, nhess, npsi, nprox` | table columns | `Counters` fields |

---

## Part XIV — "Pretend you wrote it": defense Q&A

The questions you should be able to answer cold.

**Q. What problem does this solve and why is it hard?**
`min f(x) + suppA(F(x)) + phi(x)` — smooth + risk + prox-friendly convex. Hard
because it's nonsmooth *and* nonconvex *and* `F` may be expensive (a PDE per
eval). Standard smooth methods can't handle the nonsmoothness; standard nonsmooth
methods can't afford expensive `F`. (`main.tex:147`)

**Q. Why a trust region instead of a proximal gradient?**
Trust regions handle nonconvexity gracefully (a bad model just gets rejected and
the radius shrinks) and — crucially — let you control *inexactness* via the
acceptance test (Conditions 1 & 2). A plain prox-gradient has no comparable
mechanism for cheap-but-approximate `F`.

**Q. Where does the risk actually enter?**
`suppA` is the support function of the risk envelope `A`; for a coherent risk
measure, `R(F(x)) = suppA(F(x))`. The envelope type (`IntervalEnvelope`,
`ConvexCombinationEnvelope`) *is* the choice of risk measure.

**Q. Why is there an inner algorithm at all?**
Because `psi_k = suppA∘ℓ_k + phi` has no closed-form prox. Algorithm 2 computes it
by maximizing the concave dual `d(θ)` over `θ ∈ A` with a nonmonotone BB spectral
projected gradient, then recovering the primal `p(θ̄)`. (`main.tex:687`)

**Q. Why the explicit `HilbertSpace`? Julia already has `dot`.**
`dot` is the Euclidean metric `M = I` — the *wrong* inner product for the
function-space problems this targets. Gradients are dual objects; you need the
Riesz map `M⁻¹` to step in the primal space; and **mesh independence only holds in
the mass-matrix metric**. Euclidean is kept as the zero-overhead `Nothing` special
case. (Part III.2; `main.tex:1675`)

**Q. What's the function barrier and why does it matter?**
`solve` builds the concrete `psi`/`model` types, then dispatches once into a typed
kernel (`_solve_exact!` / `_solve_inexact!`). Without it, the hot loop would
dynamically dispatch on a union-typed model every iteration — slow and
allocation-heavy. (`trustregion.jl:364`)

**Q. How do you know `h_k` is trustworthy?**
`hstat!` returns an `ok` flag from the inner prox. The solver only certifies
convergence when `h_k ≤ htol` **and** `ok`. If the prox capped out, the status is
`:prox_failure`, not `:converged`. (`trustregion.jl:285`)

**Q. What did you port from ROL and what's new?**
Ported: the TypeP trust-region scaffold (`dcauchy`/`dspg2`/`dncg`/`dbls`, the
`run` loop, `analyzeRatio`, the FD checks, the prox formulas). New: the risk
envelopes, Algorithm 2's dual prox, the inexact oracle, the metric generalization,
and the Julia engineering (function barrier, zero-alloc kernels, generic `T`).
~half/half by lines; the new half is what the paper invented. (Part VII)

**Q. Does it reproduce the paper's results?**
No — and that's by design. The paper's examples are PDE-constrained (Burgers SAA,
semilinear elliptic FEM); this is a finite-dimensional proof of concept. It
validates the *theory and structure* (the `h_k` criterion, Conditions 1 & 2,
closed-form optima, mesh-consistent metrics), not the PDE tables. (Part XII)

**Q. What's the single most important correctness invariant?**
The true objective always uses the exact `F`, never the linearization `ℓ_k`. The
model and the truth are separate quantities; conflating them would make the ratio
test meaningless. (`trustregion.jl:138`, `test_risk_finitedim_p1.jl:65`)

---

## Part XV — Where to go next

- **Read the code in include order** (Part III.4), with this guide open beside it.
- **Run the demo generator** to watch real solves produce real numbers:
  `julia --project demo/generate_evidence.jl`.
- **Read `test_risk_finitedim_p1.jl`** as a worked, asserted example.
- **Skim the paper** in this order: §1 intro (`main.tex:147`), Algorithm 1
  (`:591`), Algorithm 2 (`:687`), then the numerics tables (`:1522`, `:1646`) to
  see what "done" looks like.
- **If you want the real paper comparison:** implement the §5.2 semilinear elliptic
  `F` with a FEM discretization, wire its adjoint into `VectorMap`, and run a
  mesh-refinement sweep. That is the one piece standing between "faithful
  finite-dim port" and "reproduces the paper."

---

*Built from a full read of all 10 source files, the test suite, the demo, and the
paper. Every code pointer is `file.jl:line`; every paper pointer is
`main.tex:line` against the arXiv:2604.07216v1 source.*
