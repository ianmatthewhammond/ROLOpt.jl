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

# Cauchy-search helpers, hoisted to module scope (NOT closures) so `dcauchy!`
# infers a concrete return: capturing these locals in nested closures previously
# boxed sval/nval/snorm (inferred `Tuple{Float64,Any,Any,Any,Float64}`). Each
# returns this step's dual-SPG residual; the caller folds it into a plain `T`
# running maximum (no `Ref`, so the kernels stay allocation-free).

# One proximal-gradient step; counts it. Returns this step's dual-SPG residual.
@inline function _cauchy_pgs!(s, px, nobj, x, dg, alpha, cnt::Counters, ::Type{T}) where {T}
    res = pgstep!(px, s, nobj, x, dg, alpha)
    countprox!(cnt, res)
    return spg_pgnorm(res, T)
end

# Model decrease at `alpha`. Writes s, px; returns (sval,nval,pRed,Qk,snorm,wpg).
function _cauchy_trystep!(s, px, dwa, model, nobj, X::HilbertSpace{T}, x, dg, alpha,
                          sold::T, nold::T, half::T, cnt::Counters) where {T}
    wpg = _cauchy_pgs!(s, px, nobj, x, dg, alpha, cnt, T)
    snorm = spnorm(X, s)
    modelhessvec!(dwa, model, s); cnt.nhess += 1
    nval = value(nobj, px); cnt.npsi += 1
    gs = inner(X, s, dg)
    sval = sold + gs + half * apply(dwa, s)
    pRed = (sold + nold) - (sval + nval)
    Qk = gs + nval - nold
    return sval, nval, pRed, Qk, snorm, wpg
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
    worst = zero(T)                 # worst dual-SPG residual over all proxes here

    # Initial step
    worst = max(worst, _cauchy_pgs!(s, px, nobj, x, dg, alpha, cnt, T))
    snorm = spnorm(X, s)
    sval = sold; nval = nold; pRed = zero(T); Qk = zero(T)
    interp = snorm > del
    if !interp
        sval, nval, pRed, Qk, snorm, wpg =
            _cauchy_trystep!(s, px, dwa, model, nobj, X, x, dg, alpha, sold, nold, half, cnt)
        worst = max(worst, wpg)
        interp = pRed < -p.mu0 * Qk
    end

    cnt_steps = 0
    if interp                       # decrease alpha until sufficient model decrease
        search = true
        while search
            alpha *= p.interpf
            worst = max(worst, _cauchy_pgs!(s, px, nobj, x, dg, alpha, cnt, T))
            snorm = spnorm(X, s)
            if snorm <= del
                sval, nval, pRed, Qk, snorm, wpg =
                    _cauchy_trystep!(s, px, dwa, model, nobj, X, x, dg, alpha, sold, nold, half, cnt)
                worst = max(worst, wpg)
                search = (pRed < -p.mu0 * Qk) && (cnt_steps < p.redlim)
            end
            cnt_steps += 1
        end
    else                            # extrapolate: increase alpha while it helps
        alphas = alpha; mvals = pRed
        copyto!(dwa1, dwa)
        search = true
        while search
            alpha *= p.extrapf
            worst = max(worst, _cauchy_pgs!(s, px, nobj, x, dg, alpha, cnt, T))
            snorm = spnorm(X, s)
            if snorm <= del && cnt_steps < p.explim
                sval, nval, pRed, Qk, snorm, wpg =
                    _cauchy_trystep!(s, px, dwa, model, nobj, X, x, dg, alpha, sold, nold, half, cnt)
                worst = max(worst, wpg)
                if pRed >= -p.mu0 * Qk && abs(pRed - mvals) > p.qtol * abs(mvals)
                    copyto!(dwa1, dwa); alphas = alpha; mvals = pRed
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
    worst = max(worst, _cauchy_pgs!(s, px, nobj, x, dg, alpha, cnt, T))
    snorm = spnorm(X, s)
    modelhessvec!(dwa, model, s); cnt.nhess += 1
    nval = value(nobj, px); cnt.npsi += 1
    gs = inner(X, s, dg)
    sval = sold + gs + half * apply(dwa, s)
    return alpha, sval, nval, snorm, worst
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
    worst = zero(T)                 # worst dual-SPG residual over all proxes here (no Ref => alloc-free)

    # Model gradient at y (coordinate): gmod = ∇f(x_k) + H (y - x_k) = ∇q_k(y).
    @. dx = y - model.xk
    modelhessvec!(hs, model, dx); cnt.nhess += 1
    @. gmodc = gfk + hs
    nold = nval

    # Initial spectral step + proximal-gradient step from y.
    riesz!(gmodp, X, gmodc)
    lambda = clamp(one(T) / max(spnorm(X, gmodp), eps(T)), p.lambdaMin, p.lambdaMax)
    res = pgstep!(pgi, step, nobj, y, gmodp, lambda)
    countprox!(cnt, res); worst = max(worst, spg_pgnorm(res, T))
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
        countprox!(cnt, res); worst = max(worst, spg_pgnorm(res, T))
        gs = apply(gmodc, step)
        ss = inner(X, step, step)
        sqrt(ss) / lambda <= gtol && break               # proximal-gradient stationary
    end
    return sval, nval, worst
end

"""Nonlinear-CG improver parameters (defaults from ROL). `ncgType` selects the
β formula (0:FR 1:PR+ 2:HZ 3:HS+ 4:DY+ 5:FR-PR 6:DY-HS); default 4 (Dai–Yuan+)."""
struct NcgParams{T}
    t0::T; lambdaMin::T; lambdaMax::T; tol1::T; tol2::T
    maxit::Int; ncgType::Int; etaNCG::T; desPar::T
end
function NcgParams{T}(; t0=1, lambdaMin=1e-8, lambdaMax=1e8, tol1=1e-4, tol2=1e-2,
                      maxit=25, ncgType=4, etaNCG=1e-2, desPar=0.2) where {T}
    0 <= ncgType <= 6 || throw(ArgumentError("NcgParams ncgType must be in 0:6"))
    NcgParams{T}(T(t0), T(lambdaMin), T(lambdaMax), T(tol1), T(tol2),
                 Int(maxit), Int(ncgType), T(etaNCG), T(desPar))
end

"""
    dbls!(y, s, lambda, tmax, kappa, gs, nobj, pwa, nval0, cnt) -> (alpha, nval)

1-D minimizer of the TR model along `s`: `t -> ½t²·kappa + t·gs + nobj(y + t·s)`,
over `t in [0, tmax]`, by Brent's method (quadratic interpolation + golden section).
`nobj` is `psi_k` (the full nonsmooth term); only `value(nobj, ·)` is evaluated.
`nval0` is the nonsmooth value at `y`. Port of ROL `TrustRegionAlgorithm::dbls`.
"""
function dbls!(y, s, lambda, tmax, kappa, gs, nobj, pwa, nval0::T, cnt) where {T}
    tol0 = T(1e4) * sqrt(eps(T))
    epsb = T(1e-2) * sqrt(eps(T))
    eps0 = T(1e2) * eps(T)
    half = T(1) / 2; two = T(2)
    lam = (T(3) - sqrt(T(5))) / 2                         # golden-section parameter
    mu = T(1e-4)
    nold = nval0
    # Empty interval (tmax <= 0, e.g. the Cauchy point already on the TR boundary):
    # no feasible positive step. Return early — otherwise t0 = tR = 0 below makes the
    # quadratic-upper-bound minimizer ((nold-n0)/kappa)/t0 a 0/0 NaN.
    tmax <= zero(T) && return zero(T), nold
    tL = zero(T); pL = zero(T)
    tR = tmax
    @. pwa = y + tR * s; nR = value(nobj, pwa); cnt.npsi += 1
    pR = tR * (half * tR * kappa + gs) + nR - nold
    # minimizer of the quadratic upper bound
    t0 = tR; n0 = nR
    if tmax > lambda
        t0 = lambda
        @. pwa = y + t0 * s; n0 = value(nobj, pwa); cnt.npsi += 1
    end
    ts = kappa > zero(T) ? min(tR, ((nold - n0) / kappa) / t0 - gs / kappa) : tR
    t = min(t0, ts)
    useOptT = true
    if t <= tL
        t = t0 < tR ? t0 : half * tR
        useOptT = false
    end
    @. pwa = y + t * s; nt = value(nobj, pwa); cnt.npsi += 1
    Qt = t * gs + nt - nold
    pt = half * t * t * kappa + Qt
    # phi constant along s => quadratic model: use the analytic minimizer
    if useOptT && nt == nold && nR == nold
        return ts, nold
    end
    # concave model => minimum at an end point
    if pt >= max(pL, pR)
        return tR, nR
    end
    # Brent's method
    w = t; v = t; pw = pt; pv = pt
    d = zero(T); e = zero(T); u = zero(T)
    tm = half * (tL + tR)
    tol1 = tol0 * abs(t) + epsb
    tol2 = two * tol1
    for _ in 1:50
        dL = tL - t; dR = tR - t
        if abs(e) > tol1
            r = (t - w) * (pt - pv)
            q = (t - v) * (pt - pw)
            p = (t - v) * q - (t - w) * r
            q = two * (q - r)
            q > zero(T) && (p = -p)
            q = abs(q)
            etmp = e; e = d
            if abs(p) >= abs(half * q * etmp) || p <= q * dL || p >= q * dR
                e = t > tm ? dL : dR
                d = lam * e
            else
                d = p / q
                u = t + d
                (u - tL < tol2 || tR - u < tol2) && (d = tm >= t ? tol1 : -tol1)
            end
        else
            e = t > tm ? dL : dR
            d = lam * e
        end
        u = t + (abs(d) >= tol1 ? d : (d >= zero(T) ? tol1 : -tol1))
        @. pwa = y + u * s; nu = value(nobj, pwa); cnt.npsi += 1
        Qu = u * gs + nu - nold
        pu = half * u * u * kappa + Qu
        if pu <= pt
            u >= t ? (tL = t) : (tR = t)
            v = w; w = t; t = u
            pv = pw; pw = pt; pt = pu
            nt = nu; Qt = Qu
        else
            u < t ? (tL = u) : (tR = u)
            if pu <= pw || w == t
                v = w; w = u; pv = pw; pw = pu
            elseif pu <= pv || v == t || v == w
                v = u; pv = pu
            end
        end
        tm = half * (tL + tR)
        tol1 = tol0 * abs(t) + epsb
        tol2 = two * tol1
        tol3 = eps0 * max(abs(Qt), one(T))
        (pt <= mu * min(zero(T), Qt) + tol3 && abs(t - tm) <= tol2 - half * (tR - tL)) && break
    end
    return t, nt
end

# NCG beta (the 7 ROL ncgType formulas). `g2` = current "negative gradient" (the
# scaled prox-grad step), `g4` = previous one; `diff` is scratch (set to g4 - g2).
@inline function _ncg_beta(ncgType, X::HilbertSpace{T}, s, g2, g4, diff, gnorm2, gnorm0, etaNCG) where {T}
    two = T(2); g0sq = gnorm0 * gnorm0
    ncgType == 0 && return gnorm2 / g0sq                 # Fletcher–Reeves
    @. diff = g4 - g2                                    # y_k = g_{k-1} - g_k
    if ncgType == 1                                      # Polak–Ribière+
        return max(zero(T), -inner(X, diff, g2) / g0sq)
    elseif ncgType == 2                                  # Hager–Zhang
        sy = inner(X, s, diff)
        gg = inner(X, g2, g4)
        eta = -one(T) / (spnorm(X, s) * min(etaNCG, gnorm0))
        return max(eta, (gnorm2 - gg - two * inner(X, g2, s) * (gnorm2 - two * gg + g0sq) / sy) / sy)
    elseif ncgType == 3                                  # Hestenes–Stiefel+
        return max(zero(T), -inner(X, g2, diff) / inner(X, s, diff))
    elseif ncgType == 4                                  # Dai–Yuan+
        return max(zero(T), gnorm2 / inner(X, s, diff))
    elseif ncgType == 5                                  # FR–PR hybrid
        return max(-gnorm2, min(gnorm2, -inner(X, diff, g2))) / g0sq
    else                                                 # ncgType == 6: DY–HS hybrid
        return max(zero(T), min(-inner(X, g2, diff), gnorm2) / inner(X, s, diff))
    end
end

"""
    dncg!(y, model, nobj, X, x, gfk, del, sval, nval, s, pwa1, pwa2, pwa3, pwa4, pwa5, dwa, gmodc, gmodp, cnt, p) -> (sval, nval, worst_pgnorm)

Nonlinear-CG improver. Starting from the Cauchy point `y` (= x + s_c) with model
values `sval`/`nval`, take prox-gradient/NCG steps (each step length from a 1-D
Brent line search `dbls!`, capped at the trust-region boundary) to further reduce
the model `m_k(y) = q_k(y) + psi_k(y)` within `‖y-x‖_X ≤ del`. The descent/restart
test keeps the model non-increasing, so the fraction-of-Cauchy-decrease still
holds. `gfk` is the coordinate gradient `∇f(x_k)`; `worst_pgnorm` is the worst
dual-SPG residual among the proxes here. Port of ROL `TrustRegionAlgorithm::dncg`.
(`s/pwa1..5/dwa/gmodc/gmodp` are scratch.)
"""
function dncg!(y, model, nobj, X::HilbertSpace{T}, x, gfk, del,
               sval, nval, s, pwa1, pwa2, pwa3, pwa4, pwa5, dwa, gmodc, gmodp, cnt, p) where {T}
    half = T(1) / 2; two = T(2)
    del2 = del * del
    safeguard = sqrt(eps(T))
    worst = zero(T)
    nold = nval

    # Model gradient at y (coordinate): gmodc = ∇f(x_k) + H (y - x_k).
    @. pwa3 = y - model.xk
    modelhessvec!(dwa, model, pwa3); cnt.nhess += 1
    @. gmodc = gfk + dwa
    @. pwa3 = y - x                                      # running y - x (Cauchy step so far)
    ss0 = inner(X, pwa3, pwa3)
    snorm0 = sqrt(ss0)

    # Initial spectral step length + prox-gradient (Cauchy) direction from y.
    riesz!(gmodp, X, gmodc)
    lambda = clamp(p.t0 / max(spnorm(X, gmodp), eps(T)), p.lambdaMin, p.lambdaMax)
    lam1 = lambda
    res = pgstep!(pwa1, pwa2, nobj, y, gmodp, lambda)
    countprox!(cnt, res); worst = max(worst, spg_pgnorm(res, T))
    @. pwa2 = pwa2 / lambda                              # (prox - y)/lambda
    @. s = pwa2
    gs = apply(gmodc, s)
    snorm = spnorm(X, s)
    gnorm = snorm
    gtol = min(p.tol1, p.tol2 * gnorm)

    for _ in 1:p.maxit
        modelhessvec!(dwa, model, s); cnt.nhess += 1     # H·s (coord)
        sHs = apply(dwa, s)
        ss = snorm * snorm
        ds = inner(X, s, pwa3)
        # Largest alpha keeping ‖(y-x) + alpha·s‖_X ≤ del, then the 1-D model min.
        # Clamp to >= 0 (the Cauchy point can already sit on the boundary, giving a
        # degenerate/negative root) and guard ss = 0 (a collapsed direction).
        disc = ds * ds + ss * (del2 - snorm0 * snorm0)
        alphaMax = ss > zero(T) ? max(zero(T), (-ds + sqrt(max(disc, zero(T)))) / ss) : zero(T)
        # alphaMax == 0 means snorm0 >= del (already at the boundary) and s points
        # outward: no feasible step, so we are done (on the TR boundary).
        alphaMax <= zero(T) && break
        alpha, nold = dbls!(y, s, lam1, alphaMax, sHs, gs, nobj, pwa5, nold, cnt)

        @. y += alpha * s
        @. gmodc += alpha * dwa
        sval += alpha * (gs + half * alpha * sHs)
        @. pwa3 = y - x
        ss0 += alpha * (alpha * ss + two * ds)
        snorm0 = sqrt(max(ss0, zero(T)))
        snorm0 >= (one(T) - safeguard) * del && break    # on the TR boundary

        riesz!(gmodp, X, gmodc)
        lambda = clamp((sHs <= safeguard) ? p.t0 / max(spnorm(X, gmodp), eps(T)) : ss / sHs,
                       p.lambdaMin, p.lambdaMax)
        @. pwa4 = pwa2                                   # previous "negative gradient"
        res = pgstep!(pwa1, pwa2, nobj, y, gmodp, lambda)
        countprox!(cnt, res); worst = max(worst, spg_pgnorm(res, T))
        @. pwa2 = pwa2 / lambda
        gnorm0 = gnorm
        gnorm = spnorm(X, pwa2)
        gnorm <= gtol && break                           # prox-gradient stationary

        gnorm2 = gnorm * gnorm
        beta = _ncg_beta(p.ncgType, X, s, pwa2, pwa4, pwa5, gnorm2, gnorm0, p.etaNCG)
        reset = true
        if beta != zero(T) && isfinite(beta)
            @. pwa5 = pwa2 + beta * s                    # candidate CG direction
            @. pwa4 = y + pwa5
            nval = value(nobj, pwa4); cnt.npsi += 1
            gs = apply(gmodc, pwa5)
            if gs + nval - nold <= -(one(T) - p.desPar) * gnorm2
                @. s = pwa5; lam1 = one(T); reset = false
            end
        end
        if reset                                         # beta=0 or not a descent dir
            @. s = pwa2
            gs = apply(gmodc, s)
            lam1 = lambda
        end
        snorm = spnorm(X, s)
    end
    return sval, nold, worst
end
