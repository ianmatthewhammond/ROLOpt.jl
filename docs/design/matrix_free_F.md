# Design: user-controlled `F` representation (explicit / cached-Jacobian / matrix-free)

**Status:** design / proposal (no code yet). For review by Ian + Codex.
**Scope:** let the user choose how the risk map `F` is represented; keep the
explicit path unchanged; add a matrix-free path and an assemble-once-per-anchor
path. The solver loop and Algorithm 2 are otherwise unchanged.

## Context & guiding principle

The risk map `F : X → Y` in `min f(x) + suppA(F(x)) + phi(x)` is today always an
**explicit fixed matrix** (`AffineMap`/`QuadraticMap`). PDE-constrained problems
need more: at scale **N = dim X ≈ 1M, M = dim Y ≈ 10M** a dense reduced Jacobian
cannot exist, so `F` must be matrix-free (operator actions = tangent/adjoint
solves); at smaller scale a nonlinear `F` may want its sparse Jacobian assembled
once per outer iteration and reused as gemvs.

**Guiding principle: ROLOpt honors the representation the user chose; it does not
guess whether `A_k` should exist.** The user controls the representation by the
`VectorMap` type they pass. **No automatic dense assembly, no `min(M,N)` probing in
`solve`** (dangerous at PDE scale; a probing utility may exist later as an opt-in
helper, never solver behavior).

## Core: `VectorMap` stays the only solver-facing abstraction

The solver and Algorithm 2 use **only** `value!`, `applyJacobian!`,
`applyAdjointJacobian!`, optional `applyAdjointHessian!`, `rangedim`
(`src/objectives.jl:79-97`), all applied at the frozen `psi.xk`
(`psimodel.jl:129,140,145`; anchor set only by `reanchor!`). We add **one**
interface method (`prepare!`) and **two** map types.

### New interface method: `prepare!(F, xk)`

```julia
prepare!(F::VectorMap, x) = F            # default no-op (AffineMap/QuadraticMap inherit)
```
`reanchor!` calls it once per anchor, **before** `value!`:
```julia
function reanchor!(psi::PsiModel, xk)
    copyto!(psi.xk, xk)
    prepare!(psi.F, psi.xk)        # NEW: PDE factorization / Jacobian assembly hook
    value!(psi.bk, psi.F, xk)
    return psi
end
```
- The single "anchor moved" hook: a PDE map factorizes `K(x_k)`; a cached-Jacobian
  map assembles `A_k`. `psi.F` and the curvature `model.F` are the **same object**,
  so this one call readies both the Jacobian (Algorithm 2) and `F''`
  (`modelhessvec!`) caches.
- Forwarding: `prepare!(im::InexactMap, x) = (prepare!(im.F, x); im)`.
- Export `prepare!`.

### Cache contract (value cache vs linearization cache)

`prob.F` is used for **two distinct things**: the frozen model map inside
`PsiModel` (`applyJacobian!`/`applyAdjointJacobian!` at `x_k`), and
`objective_value!(prob.F, xtrial)` at **trial** points for the ratio test. For a PDE
map, `value!(F, xtrial)` may solve/factor/cache at `xtrial`. **That must not corrupt
the `prepare!`-built linearization at `x_k`** — because after a *rejected* step the
solver keeps `x_k` and reuses the `x_k` Jacobian (the exact-reject path shrinks `Δ`
and loops **without** re-`reanchor!`, `trustregion.jl:295-296`).

**Contract (the design):**
- `prepare!(F, x_k)` **owns** the linearization cache (factorization / assembled
  `A_k`), keyed to `x_k`.
- `value!(F, x)` may keep its **own separate** transient state but **must not mutate
  the linearization cache**. A `value!` at a rejected trial leaves the `x_k`
  Jacobian intact.
- Document for all map authors (OperatorMap / CachedJacobianMap / subtypes).
- **Adversarial test (required):** a map whose `value!(xtrial)` deliberately writes a
  *separate* value cache; run a problem that produces **rejections**; verify the
  trajectory still matches the explicit oracle.

So **"prepare! once per anchor" is precise only for accepted anchors.** Cache
separation makes that sufficient. *Fallback* (only if a map cannot separate):
re-call `prepare!(psi.F, psi.xk)` after an exact rejected trial — but that
re-factorizes the same `K(x_k)`, defeating caching, so the contract is preferred.
(The inexact-reject path already re-refreshes; only exact-reject does not.)

### Three user-chosen representations

1. **`AffineMap` (exists, unchanged)** — fixed `A`, `F'(x)=A` always. No `prepare!`.
2. **`OperatorMap` (new) — pure matrix-free.** `value!`, `jvp!`, `vjp!`, optional
   `ahp!`, optional `prepare!`. No stored `A_k`; every apply is a user callback
   (PDE solve). For `F` whose Jacobian cannot be stored.
3. **`CachedJacobianMap` (new) — assemble-once-per-anchor, gemv applies.** Nonlinear
   `F`, `A_k = F'(x_k)`; user supplies `assemble!(Ak, x)`. `prepare!` runs
   `assemble!`; applies are cached gemvs; rebuilt when `x_k` changes. **Distinct from
   `AffineMap`** (fixed `A` vs local linearization rebuilt per anchor).

```julia
# (2) pure matrix-free — callbacks may return nothing, so methods force the buffer return
struct OperatorMap{T,FV,FJ,FA,FH,FP} <: VectorMap{T}
    m::Int; fval::FV; fjvp::FJ; fvjp::FA; fahp::FH; fprep::FP   # fahp/fprep may be Nothing
end
OperatorMap{T}(; m, value!, jvp!, vjp!, ahp!=nothing, prepare!=nothing)
rangedim(F::OperatorMap) = F.m
value!(b,F,x)               = (F.fval(b,x); b)
applyJacobian!(o,F,v,x)     = (F.fjvp(o,v,x); o)               # primal Y
applyAdjointJacobian!(o,F,th,x) = (F.fvjp(o,th,x); o)          # coord X, bare transpose
applyAdjointHessian!(o,F,th,v,x) = F.fahp===nothing ?
    throw(ArgumentError("OperatorMap has no ahp!; set use_curvature=false or supply ahp!")) :
    (F.fahp(o,th,v,x); o)
prepare!(F::OperatorMap,x)  = (F.fprep===nothing ? nothing : F.fprep(x); F)

# (3) assemble-and-cache the local Jacobian at x_k; assemble! may mutate Ak or return a new one
mutable struct CachedJacobianMap{T,MA<:AbstractMatrix{T},FV,FAS,FH} <: VectorMap{T}
    m::Int; Ak::MA; fval::FV; fassemble::FAS; fahp::FH
end
CachedJacobianMap{T}(; m, Ak, value!, assemble!, ahp!=nothing)
rangedim(F::CachedJacobianMap) = F.m
function prepare!(F::CachedJacobianMap, x)
    newAk = F.fassemble(F.Ak, x)            # mutate Ak in place, or return a new (same-type) matrix
    newAk === nothing || (F.Ak = newAk)
    return F
end
value!(b,F,x)           = (F.fval(b,x); b)                 # true nonlinear F(x) (for b_k)
applyJacobian!(o,F,v,x) = (mul!(o, F.Ak, v); o)            # cached gemv
applyAdjointJacobian!(o,F,th,x) = (mul!(o, transpose(F.Ak), th); o)
applyAdjointHessian!(o,F,th,v,x) = F.fahp===nothing ?     # mirrors OperatorMap: hard error
    throw(ArgumentError("CachedJacobianMap has no ahp!; set use_curvature=false or supply ahp!")) :
    (F.fahp(o,th,v,x); o)
```
`CachedJacobianMap.value!` is the *true* `F` (for `b_k`); `A_k` is the linearization,
never derived by ROLOpt.

### Conventions the user MUST respect (the #1 pitfall — document loudly)

`value!`/`applyJacobian!` return **primal Y** (`F(x)`, `A v`);
`applyAdjointJacobian!` returns the **coordinate X** vector `Aᵀθ` — the *bare
transpose, no metric*. The framework forms the metric adjoint
`A_k^*θ = riesz_X(Aᵀ(M_Y θ))` itself (`psimodel.jl:139-141`). So the VJP is the plain
transpose-Jacobian action, consistent with the JVP, so `⟨A v, θ⟩ = ⟨v, Aᵀθ⟩` holds
(what `check_adjoint` tests). PDE: **discrete adjoint = transpose of the discrete
tangent, in coordinates**.

### `prepare!` in the derivative checks

`check_jacobian(F, x, v)` and `check_adjoint(F, x, v, theta)` (`src/check.jl`) must
call `prepare!(F, x)` before applying the Jacobian — otherwise `CachedJacobianMap`
has no assembled `A_k` and validation is stale.

**Opt-in solve-start validation (Phase 1).** `TROptions.validate_F` (default
`false`). When `true`, `solve` runs a deterministic adjoint-consistency **smoke
check** at the start and throws on a mis-wired VJP. It is **deterministic (no
`Random` in `src`)**: a few fixed probe pairs `(v, θ)` from `inexact.jl`'s
`_dir(T, k, shift)` helper (shifts `(1,2),(2,0),(0,1)`), failing when the worst
**relative** adjoint residual `|⟨A v,θ⟩ − ⟨v,Aᵀθ⟩| / (|⟨A v,θ⟩| + 1)` exceeds
`validate_tol` (a `TROptions` field, default `1e3·sqrt(eps(T))` — loose enough to
pass a correct VJP at any precision, tight enough to catch a transpose mismatch,
which is `O(1)` relative). It runs `prepare!(F, x0)` first, so it works for
`CachedJacobianMap`. **It is a guard, not a proof** — an error exactly orthogonal to
every probe would slip; for a full check the user runs `check_adjoint`/
`check_jacobian` on their own probes.

## Phase 2 — Line-search operator reuse (matrix-free ONLY; gated, secondary)

Performance for the matrix-free representation (each apply = a solve). **Gated** so
the explicit/cached paths are untouched.

**Resolved-once gating.** `prox!`/`dncg!` see only `psi`/`DualSPGOptions`, not
`TROptions`. So `solve` resolves `matrix_free_reuse ∈ {:auto,:on,:off}` (new
`TROptions` field, default `:auto`) against the trait
`operator_actions_expensive(typeof(F))` **once**, into a concrete **singleton-typed
reuse-mode field carried by `PsiModel`** (a type parameter, e.g.
`ReuseOn()`/`ReuseOff()`). `prox!`/`dncg!` dispatch on it at **compile time**, so an
explicit/cached `PsiModel` (`ReuseOff`) elides the reuse branch entirely
(`test_perf.jl` zero-alloc/`@inferred` guards hold; no runtime Symbol branch).
Trait: `operator_actions_expensive(::Type{<:VectorMap})=false`, `=true` for
`OperatorMap`. **Wrapper forwarding (required):**
`operator_actions_expensive(::Type{<:InexactMap{T,FM}}) where {T,FM} =
operator_actions_expensive(FM)` — so a matrix-free `F` wrapped as
`InexactMap(prob.F)` under inexactness keeps reuse enabled instead of silently
falling back to the no-reuse path.

**Principle.** Where an inner loop moves along a **fixed direction `s`** and applies
`A_k` to an argument **affine in the step parameter**, apply `A_k` to the pieces
ONCE and AXPY per trial instead of re-applying (a solve) per trial.

**(a) `dbls!` — NCG Brent line search (biggest win).** `subsolvers.jl:291`, from
`dncg!:463`. Each trial evaluates `value(psi, y+t·s)` = one **forward JVP**, no prox
(`psimodel.jl:126`). `ℓ_k(y+t·s) = ℓ_k(y) + t·(A_k·s)` is affine ⇒ precompute the
**primal-Y** vectors `ℓ_y = ℓ_k(y)` and `A_k·s = applyJacobian!(·,F,s,x_k)` ONCE;
each of up to ~50 Brent trials is `suppA(psi.A, psi.Y, ℓ_y + t·(A_k·s)) +
phi(y+t·s)` — **zero forward JVPs** (`suppA` applies the Y metric internally). The
curvature stays the precomputed scalar `kappa=sHs` (`dncg!:452`); **no**
`modelhessvec!`/`F''` per trial.

**(b) Dual-SPG line search.** `prox!`/`_dual_d!` (`psimodel.jl:137`, loop `:189`),
`s = ws.s` fixed. The metric adjoint `A_k^*(·) = riesz_X(Aᵀ(M_Y ·))` is **linear**,
so `A_k^*(θ+λs) = A_k^*θ + λ·A_k^*s`, where **`A_k^*s = riesz_X(Aᵀ(M_Y s))`** (a
primal-X vector — *not* the forward `A_k s`). Precompute `aθ = A_k^*θ`, `as = A_k^*s`
once per SPG iteration; per trial `adjp = aθ + λ·as` (AXPY, primal X) — saves one
**adjoint solve per trial**. On accept `aθ ← aθ + λ*·as` (free) ⇒ ~1 adjoint
solve/iteration. The forward JVP `A_k(p−x_k)` with `p = prox(z − t·adjp)` is
**nonlinear in λ** — recompute every trial (unavoidable).

**Do NOT touch:** `dcauchy!` (bidirectional; prox arg + step length both vary, prox
nonlinear); `dspg!` (closed-form 1-D step); the explicit/cached path.

**Kernel refactors (additive, gated):** split `_dual_d!` into adjoint-precompute +
`_dual_d_from_adj!`; the matrix-free `prox!` variant maintains `aθ`/`as` in new
`PsiWorkspace` buffers. `dncg!` (matrix-free) precomputes `ℓ_y`/`A_k·s` and hands
`dbls!` an affine trial-evaluator; `dbls!`'s Brent logic is otherwise unchanged.

## Files

```
src/usermaps.jl      NEW   OperatorMap + CachedJacobianMap + operator_actions_expensive trait
src/objectives.jl    EDIT  prepare!(F::VectorMap,x)=F default
src/psimodel.jl      EDIT  reanchor! calls prepare! ; (Ph2) reuse-mode param + _dual_d! split + matrix-free prox!
src/inexact.jl       EDIT  prepare!(im::InexactMap,x)=(prepare!(im.F,x);im)
src/check.jl         EDIT  check_jacobian/check_adjoint call prepare!(F,x) first
src/subsolvers.jl    EDIT  (Ph2) dbls!/dncg! affine trial-evaluator (gated)
src/trustregion.jl   EDIT  (Ph1) validate_F field + solve-start check_adjoint ; (Ph2) matrix_free_reuse + resolve into PsiModel
src/ROLOpt.jl        EDIT  export OperatorMap, CachedJacobianMap, prepare!
GUIDE.md             EDIT  "Choosing an F representation" + conventions + cache contract + reuse
test/test_matfree.jl NEW   three-mode equivalence + prepare!/cache + validation (+ Ph2 reuse counters)
test/runtests.jl     EDIT  include test_matfree.jl
README.md            EDIT  three F representations
```
**Phase 1** solver-side edit is only the no-op `prepare!` call in `reanchor!` (+
`check.jl`); `prox!`/`dncg!` untouched. **Phase 2** edits are gated; explicit/cached
paths stay byte-for-byte.

## Correctness testing (`test/test_matfree.jl`)

1. **Three-mode equivalence.** Same `F` as (A) explicit, (B) `OperatorMap`, (C)
   `CachedJacobianMap`. Linear `F`: A ≡ B bit-identical, A ≡ C within tol.
   **Nonlinear `F`** (`QuadraticMap` oracle): A ≡ B ≡ C within tol (exercises
   `CachedJacobianMap` rebuilding `A_k` per anchor). Across `:cauchy/:spg/:ncg`.
2. **Weighted metric.** Repeat (1) with `X,Y = WeightedSpace`.
3. **Wrong-VJP detection.** `check_adjoint` ≈ 0 for a correct VJP; dropped transpose
   makes it large (checks now `prepare!` first).
4. **`prepare!` discipline.** Counter on an **exact** solve: `prepare!` fires once
   at the initial anchor plus once per **accepted** step (= #accepted anchors + 1),
   **not per apply**, and there is no re-assembly within an iteration. Scope this
   invariant to exact solves — the inexact path re-`reanchor!`s more often by design
   (fixed-point sweeps / reject refreshes), so its count is higher and not asserted
   here.
5. **Cache contract (adversarial).** `value!(xtrial)` mutates a separate cache; with
   **rejections**, the trajectory still matches the explicit oracle.
6. **User factorization cache.** `OperatorMap` `prepare!` LU-factors a fixed `K`,
   JVP/VJP as transpose back-subs ≡ explicit `AffineMap(K\B…)`.
7. **Curvature.** With `ahp!` ≡ `QuadraticMap` curvature; missing `ahp!` +
   `use_curvature=true` throws.
8. **Precision.** Equivalence at `Float32`/`BigFloat`.
9. **(Phase 2) Matrix-free reuse.** `matrix_free_reuse` `:on` vs `:off`: agree within
   tol and `:on` does strictly fewer operator applies (`dbls!` forward JVPs `O(1)`
   not `O(#Brent trials)`; dual-SPG adjoint applies `O(1)` not `O(#trials)`).

## Phasing

- **Phase 1 (DONE):** `prepare!` (interface + `reanchor!` + `InexactMap` +
  `check.jl`), `OperatorMap`, `CachedJacobianMap`, the `operator_actions_expensive`
  trait, `TROptions.validate_F` opt-in solve-start check, cache contract, docs,
  tests 1–8. No `prox!`/`dncg!` reuse.
- **Phase 2 (DONE):** gated matrix-free reuse in dual-SPG + NCG — `matrix_free_reuse`
  (`:auto`/`:on`/`:off`), the reuse-mode singleton resolved once into a `PsiModel`
  type parameter (`ReuseOn`/`ReuseOff`), `_dual_d!` split into `_compute_adj!` +
  `_dual_d_from_adj!`, `_trial_adj!` AXPY, the `dbls!`/`dncg!` affine `_ls_value`
  evaluator, test 9. The explicit/cached path is `ReuseOff` (compile-time elided), so
  `test_perf` zero-alloc/`@inferred` and all trajectory tests stay byte-for-byte;
  reuse cuts adjoint applies ~10–100× on iterating instances and matches the naive
  result to rounding.
- **Phase 3:** real Gridap tangent/adjoint `F` in
  `external/photonics-opt-testbed/Focusing2D`, end-to-end.

## Verification

- `julia --project -e 'using Pkg; Pkg.test()'` green incl. `test_matfree.jl`;
  existing suite (incl. `test_perf.jl`, trajectories) unchanged.
- `check_*` ≈ 0 (after `prepare!`); wrong-VJP caught; `prepare!` counter ==
  #accepted anchors; adversarial value-cache map matches oracle through rejections;
  (Ph2) reuse `:on` drops apply counts and matches `:off` within tol.

## Resolved decisions (Ian, 2026-06-22)

1. **Cached map name:** **`CachedJacobianMap`**.
2. **`OperatorMap` expensiveness:** **always expensive** —
   `operator_actions_expensive(::Type{<:OperatorMap}) = true` (so `:auto` enables
   Phase-2 reuse for `OperatorMap`).
3. **Curvature without `ahp!`:** **hard error** if `use_curvature=true`
   (`ArgumentError`), never a silent `F''=0`.
4. **Solve-start validation:** **add `TROptions.validate_F`** (default `false`);
   when `true`, `solve` runs `check_adjoint` once and errors on a bad VJP.

## Open (deferred to Phase 3)

- **Phase-3 target:** confirm `Focusing2D` (env available), or pick a simpler PDE.
