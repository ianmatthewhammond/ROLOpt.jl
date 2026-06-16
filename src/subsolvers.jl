# Trust-region subproblem building blocks. Phase 1 ships the proximal-gradient
# step, the stationarity metric, and the Cauchy point. NCG/SPG improvers
# (dncg/dspg/dbls/dprox) are Phase 2.
#
# Blueprint: ROL_TypeP_Algorithm_Def.hpp (pgstep) and
# ROL_TypeP_TrustRegionAlgorithm_Def.hpp (dcauchy).

"""Evaluation counters (columns mirror the paper's result tables)."""
mutable struct Counters
    nfval::Int     # evaluations of the (f, F) pair
    ngrad::Int     # gradient computations
    nhess::Int     # model Hessian applications
    npsi::Int      # evaluations of psi_k
    nprox::Int     # prox_{t psi_k} evaluations
    nspg::Int      # total dual spectral-projected-gradient iterations (for averaging)
    nproxfail::Int # prox_{t psi_k} evaluations that did NOT converge
end
Counters() = Counters(0, 0, 0, 0, 0, 0, 0)

# nobj is the nonsmooth term (a PsiModel or a ProxObjective). prox! returns a
# DualSPGResult for PsiModel; the .iters / .converged fields feed the counters
# and failure tracking. For a plain ProxObjective prox! returns the output
# vector (always "converged", 0 inner iterations).
spg_iters(r::DualSPGResult) = r.iters
spg_iters(::Any) = 0
spg_ok(r::DualSPGResult) = r.converged
spg_ok(::Any) = true
# Final dual projected-gradient residual (0 for an exact ProxObjective prox).
spg_pgnorm(r::DualSPGResult{T}) where {T} = r.pgnorm
spg_pgnorm(::Any, ::Type{T}) where {T} = zero(T)
spg_pgnorm(r::DualSPGResult, ::Type{T}) where {T} = T(r.pgnorm)

# Record a prox result in the counters; returns whether it converged.
@inline function countprox!(cnt::Counters, res)
    cnt.nprox += 1
    cnt.nspg += spg_iters(res)
    ok = spg_ok(res)
    ok || (cnt.nproxfail += 1)
    return ok
end

"""
    pgstep!(pgiter, step, nobj, x, dg, t) -> res

Proximal-gradient step: `pgiter = prox_{t·nobj}(x - t·dg)`, `step = pgiter - x`.
`dg` is the primal gradient. Returns the prox result (for diagnostics/counting).
"""
function pgstep!(pgiter, step, nobj, x, dg, t)
    @. step = x - t * dg
    res = prox!(pgiter, nobj, step, t)
    @. step = pgiter - x
    return res
end

"""
    hstat!(step, pgiter, nobj, X, x, dg, t0, cnt) -> (h_k, ok)

Stationarity metric `h_k = ‖prox_{t0·nobj}(x - t0·dg) - x‖_X / t0`. Records the
prox in `cnt` and returns whether the (dual-SPG) prox converged; `h_k` is
unreliable when `ok` is false.
"""
function hstat!(step, pgiter, nobj, X::HilbertSpace, x, dg, t0, cnt::Counters)
    res = pgstep!(pgiter, step, nobj, x, dg, t0)
    ok = countprox!(cnt, res)
    return spnorm(X, step) / t0, ok
end

"""
    dcauchy!(s, model, nobj, X, x, dg, del, alpha, sval0, nval0, px, dwa, dwa1, cnt, p) -> (alpha, sval, nval, snorm, worst_pgnorm)

Cauchy point for the TR subproblem `min q_k(x_k+s) + psi_k(x_k+s)` s.t. ‖s‖ ≤ del,
via a bidirectional proximal search on the step length `alpha`. Writes the Cauchy
step into `s` (and the trial iterate `x_k+s` into `px`). `sval0 = f(x_k)`,
`nval0 = psi_k(x_k)`; returned `sval`/`nval` are the model smooth/nonsmooth values
at the Cauchy point, so `pred = (sval0+nval0) - (sval+nval)`. The last return,
`worst_pgnorm`, is the largest dual-SPG residual over all proxes used here (0 for
exact proxes); the caller treats a step with `worst_pgnorm > step_prox_tol` as
grossly unreliable.

Port of ROL `TrustRegionAlgorithm::dcauchy`.
"""
function dcauchy!(s, model, nobj, X::HilbertSpace{T}, x, dg, del, alpha,
                  sval0, nval0, px, dwa, dwa1, cnt, p) where {T}
    half = T(1) / 2
    sold = sval0
    nold = nval0
    worst = Ref(zero(T))            # worst dual-SPG residual over all proxes here

    # One proximal-gradient step; counts it and tracks the worst prox residual.
    function pgs(alpha)
        res = pgstep!(px, s, nobj, x, dg, alpha)
        countprox!(cnt, res)
        w = spg_pgnorm(res, T)
        worst[] = max(worst[], w)
        return res
    end

    # Evaluate the model decrease at step length `alpha`. Writes s, px; returns
    # (sval, nval, pRed, Qk, snorm).
    function trystep(alpha)
        pgs(alpha)
        snorm = spnorm(X, s)
        modelhessvec!(dwa, model, s); cnt.nhess += 1
        nval = value(nobj, px); cnt.npsi += 1
        gs = inner(X, s, dg)
        sval = sold + gs + half * apply(dwa, s)
        pRed = (sold + nold) - (sval + nval)
        Qk = gs + nval - nold
        return sval, nval, pRed, Qk, snorm
    end

    # Initial step
    pgs(alpha)
    snorm = spnorm(X, s)
    sval = sold; nval = nold; pRed = zero(T); Qk = zero(T)
    interp = snorm > del
    if !interp
        sval, nval, pRed, Qk, snorm = trystep(alpha)
        interp = pRed < -p.mu0 * Qk
    end

    cnt_steps = 0
    if interp                       # decrease alpha until sufficient model decrease
        search = true
        while search
            alpha *= p.interpf
            pgs(alpha)
            snorm = spnorm(X, s)
            if snorm <= del
                sval, nval, pRed, Qk, snorm = trystep(alpha)
                search = (pRed < -p.mu0 * Qk) && (cnt_steps < p.redlim)
            end
            cnt_steps += 1
        end
    else                            # extrapolate: increase alpha while it helps
        alphas = alpha; mvals = pRed; svals = sval
        copyto!(dwa1, dwa)
        search = true
        while search
            alpha *= p.extrapf
            pgs(alpha)
            snorm = spnorm(X, s)
            if snorm <= del && cnt_steps < p.explim
                sval, nval, pRed, Qk, snorm = trystep(alpha)
                if pRed >= -p.mu0 * Qk && abs(pRed - mvals) > p.qtol * abs(mvals)
                    copyto!(dwa1, dwa); alphas = alpha; mvals = pRed; svals = sval
                    search = true
                else
                    search = false
                end
            else
                search = false
            end
            cnt_steps += 1
        end
        alpha = alphas
    end

    # Finalize at the chosen alpha so s, px, sval, nval are mutually consistent.
    pgs(alpha)
    snorm = spnorm(X, s)
    modelhessvec!(dwa, model, s); cnt.nhess += 1
    nval = value(nobj, px); cnt.npsi += 1
    gs = inner(X, s, dg)
    sval = sold + gs + half * apply(dwa, s)
    return alpha, sval, nval, snorm, worst[]
end

"""Spectral proximal-gradient improver parameters (defaults from ROL)."""
struct SpgParams{T}
    lambdaMin::T; lambdaMax::T; tol1::T; tol2::T; maxit::Int
end
function SpgParams{T}(; lambdaMin=1e-8, lambdaMax=1e8, tol1=1e-4, tol2=1e-2, maxit=25) where {T}
    SpgParams{T}(T(lambdaMin), T(lambdaMax), T(tol1), T(tol2), Int(maxit))
end

"""
    dspg!(y, model, nobj, X, x, gfk, del, sval, nval, gmodc, gmodp, step, pgi, hs, dx, cnt, p) -> (sval, nval, worst_pgnorm)

Spectral proximal-gradient improver. Starting from the Cauchy point `y` (= x + s_c)
with model values `sval`/`nval` there, take Barzilai–Borwein proximal-gradient steps
to further reduce the model `m_k(y) = q_k(y) + psi_k(y)` within `‖y-x‖_X ≤ del`,
never increasing it (so the fraction-of-Cauchy-decrease bound still holds). `gfk` is
the coordinate gradient `∇f(x_k)`. The trust region is enforced inline by capping
each step at the boundary (no Brent root-find). `worst_pgnorm` is the worst dual-SPG
residual among the proxes used here. Port of ROL `TrustRegionAlgorithm::dspg2`,
restarted from the Cauchy point. (`gmodc/gmodp/step/pgi/hs/dx` are scratch.)
"""
function dspg!(y, model, nobj, X::HilbertSpace{T}, x, gfk, del,
               sval, nval, gmodc, gmodp, step, pgi, hs, dx, cnt, p) where {T}
    half = T(1) / 2
    safeguard = T(1e2) * eps(T)
    del2 = del * del
    worst = Ref(zero(T))

    # Model gradient at y (coordinate): gmod = ∇f(x_k) + H (y - x_k) = ∇q_k(y).
    @. dx = y - model.xk
    modelhessvec!(hs, model, dx); cnt.nhess += 1
    @. gmodc = gfk + hs
    nold = nval

    # Initial spectral step + proximal-gradient step from y.
    riesz!(gmodp, X, gmodc)
    lambda = clamp(one(T) / max(spnorm(X, gmodp), eps(T)), p.lambdaMin, p.lambdaMax)
    res = pgstep!(pgi, step, nobj, y, gmodp, lambda)
    countprox!(cnt, res); worst[] = max(worst[], spg_pgnorm(res, T))
    gs = apply(gmodc, step)
    ss = inner(X, step, step)
    gtol = min(p.tol1, p.tol2 * sqrt(ss) / lambda)

    for _ in 1:p.maxit
        modelhessvec!(hs, model, step); cnt.nhess += 1   # H·step (coord)
        sHs = apply(hs, step)
        nval = value(nobj, pgi); cnt.npsi += 1           # psi at the prox point

        # Cap the step at the trust-region boundary: largest alpha in [0,1] with
        # ‖(y-x) + alpha·step‖_X^2 = s0s0 + 2·alpha·ss0 + alpha^2·ss <= del^2.
        @. dx = y - x
        ss0 = inner(X, dx, step)
        s0s0 = inner(X, dx, dx)
        disc = ss0 * ss0 - ss * (s0s0 - del2)
        alphaMax = (ss > safeguard && disc >= zero(T)) ?
                   clamp((-ss0 + sqrt(disc)) / ss, zero(T), one(T)) : one(T)
        # 1-D model minimizer along step, capped by the trust region.
        alpha = (sHs <= safeguard) ? alphaMax : min(alphaMax, -(gs + nval - nold) / sHs)
        alpha = clamp(alpha, zero(T), alphaMax)

        if alpha >= one(T) - safeguard                   # full step to the prox point
            copyto!(y, pgi)
            sval += gs + half * sHs
            @. gmodc += hs
        else                                             # partial step
            @. y += alpha * step
            nval = value(nobj, y); cnt.npsi += 1
            sval += alpha * (gs + half * alpha * sHs)
            @. gmodc += alpha * hs
        end
        nold = nval

        @. dx = y - x
        spnorm(X, dx) >= del - safeguard && break        # on the TR boundary

        # Current primal model gradient, then the new spectral (BB) step length and
        # the next proximal-gradient step. (riesz! must precede the fallback that
        # reads spnorm(X, gmodp), so it uses the updated gmodc, not a stale value.)
        riesz!(gmodp, X, gmodc)
        lambda = clamp((sHs <= safeguard) ? one(T) / max(spnorm(X, gmodp), eps(T)) : ss / sHs,
                       p.lambdaMin, p.lambdaMax)
        res = pgstep!(pgi, step, nobj, y, gmodp, lambda)
        countprox!(cnt, res); worst[] = max(worst[], spg_pgnorm(res, T))
        gs = apply(gmodc, step)
        ss = inner(X, step, step)
        sqrt(ss) / lambda <= gtol && break               # proximal-gradient stationary
    end
    return sval, nval, worst[]
end
