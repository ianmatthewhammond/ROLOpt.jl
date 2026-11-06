# Part of ROLOpt.jl (MIT; see LICENSE). Portions are ported from Trilinos ROL,
# Copyright (c) 2014 NTESS and the ROL contributors, BSD-3-Clause (see LICENSE-ROL).

# TR-Risk: the main loop (paper Algorithm 1). The trial step is the Cauchy point
# (`subsolver=:cauchy`), optionally improved by spectral proximal-gradient steps
# (`subsolver=:spg`) or nonlinear-CG steps (`subsolver=:ncg`). Supports inexact
# evaluations (Conditions 1 & 2, via an `InexactOracle`) and the optional
# (theta_k, F'') Lagrangian curvature term in B_k (`use_curvature=true`). The loop
# runs in a typed kernel (`_solve_exact!` / `_solve_inexact!`) behind a function
# barrier so it never dynamically dispatches on the model/psi type.
#
# Blueprint: ROL_TypeP_TrustRegionAlgorithm_Def.hpp::run.

"""
    RiskProblem(f, F, phi, A, X, Y)

The structured problem `min_x  f(x) + suppA(F(x)) + phi(x)` over `X` with risk
envelope `A ⊂ Y`. `X`, `Y` are `HilbertSpace`s (default Euclidean).
"""
struct RiskProblem{T,SO,FM,PH,RE,SX,SY}
    f::SO
    F::FM
    phi::PH
    A::RE
    X::SX
    Y::SY
end
function RiskProblem(f::SmoothObjective{T}, F::VectorMap{T}, phi::ProxObjective{T},
                     A::RiskEnvelope{T};
                     X::HilbertSpace{T}=EuclideanSpace(T),
                     Y::HilbertSpace{T}=EuclideanSpace(T)) where {T}
    RiskProblem{T,typeof(f),typeof(F),typeof(phi),typeof(A),typeof(X),typeof(Y)}(f, F, phi, A, X, Y)
end

struct CauchyParams{T}
    mu0::T; interpf::T; extrapf::T; redlim::Int; explim::Int; qtol::T
end

# Trust-region ratio flags (ROL TRUtils::ETRFlag values).
const TR_SUCCESS    = 0   # both reductions positive (or roundoff-floor success)
const TR_POSPREDNEG = 1   # actual > 0, predicted < 0 (accept the good step)
const TR_NPOSPREDPOS = 2  # actual <= 0, predicted > 0 (reject)
const TR_NPOSPREDNEG = 3  # actual <= 0, predicted < 0 (reject)
const TR_NAN        = 4   # NaN reduction (reject)

# Ratio of actual to predicted reduction with a roundoff safeguard, returning
# (rho, flag). Port of ROL `TRUtils::analyzeRatio`: adds eps = epsi·max(1,|fold|)
# to both reductions so that when both are at the roundoff floor we declare
# rho = 1 (SUCCESS), preventing near-solution stagnation from catastrophic
# cancellation in pRed; flags nonpositive/NaN reductions so the caller rejects
# (important when a failed prox produces a meaningless predicted reduction).
# Core: given the (computed) actual reduction `aRed` and predicted `pRed`.
function analyze_ratio_red(aRed::T, pRed::T, fold::T, epsi::T) where {T}
    eps = epsi * max(one(T), abs(fold))
    aRed_safe = aRed + eps
    pRed_safe = pRed + eps
    if ((abs(aRed_safe) < epsi && abs(pRed_safe) < epsi) || aRed == pRed)
        return one(T), TR_SUCCESS
    elseif isnan(aRed_safe) || isnan(pRed_safe)
        return -one(T), TR_NAN
    end
    rho = aRed_safe / pRed_safe
    if pRed_safe < zero(T) && aRed_safe > zero(T)
        return rho, TR_POSPREDNEG
    elseif aRed_safe <= zero(T) && pRed_safe > zero(T)
        return rho, TR_NPOSPREDPOS
    elseif aRed_safe <= zero(T) && pRed_safe < zero(T)
        return rho, TR_NPOSPREDNEG
    else
        return rho, TR_SUCCESS
    end
end
# Exact path: reduction is fold - ftrial.
analyze_ratio(fold::T, ftrial::T, pRed::T, epsi::T) where {T} =
    analyze_ratio_red(fold - ftrial, pRed, fold, epsi)

"""Trust-region options (defaults follow the paper's Numerics section)."""
struct TROptions{T}
    delta0::T        # initial radius (<=0 => auto: max(‖g_1‖, ‖prox-grad step‖))
    deltamax::T
    eta1::T          # acceptance threshold
    eta2::T          # radius-growth threshold
    gamma1::T        # shrink rate
    gamma3::T        # grow rate
    t0::T            # stationarity / prox-gradient parameter t_k
    alpha0::T        # initial Cauchy step length
    cauchy::CauchyParams{T}
    trsafe::T        # ratio-test roundoff safeguard scale (x machine eps)
    step_prox_tol::T # a Cauchy step whose worst prox residual exceeds this is "gross"
    htol::T          # stop when h_k <= htol
    maxit::Int
    verbose::Int
    subsolver::Symbol  # :cauchy (Cauchy only), :spg (Cauchy + SPG), or :ncg (Cauchy + NCG)
    use_curvature::Bool  # include the (theta_k, F'') curvature term in B_k
    validate_F::Bool   # run a deterministic adjoint-consistency check on F at solve start
    validate_tol::T    # relative tolerance for that check
    matrix_free_reuse::Symbol  # :auto | :on | :off — line-search operator reuse (matrix-free F)
    dualspg::DualSPGOptions{T}
    spg::SpgParams{T}
    ncg::NcgParams{T}
end
# Note `alpha0` defaults to `t0`: per the Phase-1 rule the Cauchy bidirectional
# search starts from the same prox-gradient parameter `t0` used for h_k.
function TROptions{T}(; delta0=10, deltamax=1e12, eta1=1e-4, eta2=0.5,
                      gamma1=0.25, gamma3=10, t0=1, alpha0=t0,
                      mu0=1e-2, interpf=0.1, extrapf=10, redlim=10, explim=10, qtol=1e-8,
                      trsafe=1e2, step_prox_tol=1e-6, htol=1e-8, maxit=1000, verbose=0,
                      subsolver::Symbol=:cauchy, use_curvature::Bool=false,
                      validate_F::Bool=false, validate_tol=1000 * sqrt(eps(T)),
                      matrix_free_reuse::Symbol=:auto,
                      dualspg::DualSPGOptions{T}=DualSPGOptions{T}(),
                      spg::SpgParams{T}=SpgParams{T}(),
                      ncg::NcgParams{T}=NcgParams{T}()) where {T}
    subsolver in (:cauchy, :spg, :ncg) ||
        throw(ArgumentError("subsolver must be :cauchy, :spg, or :ncg"))
    matrix_free_reuse in (:auto, :on, :off) ||
        throw(ArgumentError("matrix_free_reuse must be :auto, :on, or :off"))
    cp = CauchyParams{T}(T(mu0), T(interpf), T(extrapf), Int(redlim), Int(explim), T(qtol))
    TROptions{T}(T(delta0), T(deltamax), T(eta1), T(eta2), T(gamma1), T(gamma3),
                 T(t0), T(alpha0), cp, T(trsafe), T(step_prox_tol), T(htol),
                 Int(maxit), Int(verbose), subsolver, use_curvature,
                 validate_F, T(validate_tol), matrix_free_reuse, dualspg, spg, ncg)
end

"""
Result of `solve`. `converged` is true only when `h_k <= htol` AND the prox that
produced that `h_k` itself converged. `status` is one of `:converged`;
`:prox_failure` (the stationarity prox hit its iteration cap, so `h_k` is
unreliable); `:stalled` (the trust-region radius collapsed below the iterate
precision before reaching htol — e.g. the achievable stationarity floor is above
htol under large inexactness); or `:maxiter`.
"""
struct TRResult{T,V}
    x::V
    J::T
    hk::T
    iters::Int
    converged::Bool
    status::Symbol
    counters::Counters
end

"""
    objective_value!(Fbuf, prob, x) -> J(x)

The TRUE objective `f(x) + suppA(F(x)) + phi(x)` (uses the exact `F`, not the
linearization). Distinct from the model values `psi_k.value` / `q_k`.
"""
function objective_value!(Fbuf, prob::RiskProblem, x)
    value!(Fbuf, prob.F, x)
    return value(prob.f, x) + suppA(prob.A, prob.Y, Fbuf) + value(prob.phi, x)
end

# Preallocated working set, bundled so the typed solver kernels (below) take a
# small, stable argument list. All vectors are X-length except `Fbuf` (Y-length).
struct _SolveBuffers{T,VX<:AbstractVector{T},VY<:AbstractVector{T}}
    x::VX; xtrial::VX; gcoord::VX; dg::VX
    s::VX                    # Cauchy step
    step::VX                 # hstat step
    pgiter::VX
    px::VX                   # dcauchy trial iterate
    dwa::VX; dwa1::VX
    gmodc::VX; gmodp::VX     # SPG / NCG improver scratch
    ncgaux::VX               # NCG pwa5 scratch (the one extra buffer NCG needs)
    Fbuf::VY
end
function _SolveBuffers(::Type{T}, x0::AbstractVector{T}, m::Integer) where {T}
    n = length(x0)
    vx() = zeros(T, n)
    _SolveBuffers{T,Vector{T},Vector{T}}(copyto!(zeros(T, n), x0), vx(), vx(), vx(),
        vx(), vx(), vx(), vx(), vx(), vx(), vx(), vx(), vx(), zeros(T, m))
end

# Exact model refresh at `bw.x`: one pass, returns (h_k, hok).
function _refresh_exact!(prob::RiskProblem, opts, psi, model, bw, cnt)
    X = prob.X
    gradient!(bw.gcoord, prob.f, bw.x); cnt.ngrad += 1
    riesz!(bw.dg, X, bw.gcoord)
    setmodel!(model, bw.x, bw.gcoord)
    reanchor!(psi, bw.x)
    hk, hok = hstat!(bw.step, bw.pgiter, psi, X, bw.x, bw.dg, opts.t0, cnt)
    freeze_curvature!(model, psi.theta)                 # theta_k from the stationarity prox
    return hk, hok
end

# Inexact model refresh at `bw.x` for radius `delta`, returns (h_k, hok). A
# warm-started fixed point injects (g_k, A_k, b_k) at tolerances kappa·min(h_k, Δ[^2])
# and refines h_k, followed by a POSTCONDITION injection consistent with the final
# h_k. The postcondition guarantees the data used by the model/steps satisfies
# Condition 1 against the REPORTED h_k for any kappa (the fixed point — a
# contraction only for kappa < 1 — just improves the estimate; correctness of the
# contract does not depend on its convergence). The reported metric is the
# conservative max(hk, hpost), so it cannot falsely certify convergence.
function _refresh_inexact!(prob::RiskProblem{T}, opts, psi, model, oracle, imap, bw, hk0, delta, cnt) where {T}
    X = prob.X; Y = prob.Y
    hk = hk0; hok = false
    for _ in 1:10
        set_tols!(oracle, hk, delta)
        inject_gradient!(bw.gcoord, oracle, X, bw.x); cnt.ngrad += 1
        inject_linearization!(imap, oracle, X, Y, bw.x)
        riesz!(bw.dg, X, bw.gcoord)
        setmodel!(model, bw.x, bw.gcoord)
        reanchor!(psi, bw.x)                            # b_k = F(x) + e_b via imap
        hknew, hok = hstat!(bw.step, bw.pgiter, psi, X, bw.x, bw.dg, opts.t0, cnt)
        done = abs(hknew - hk) <= T(1e-3) * max(hknew, eps(T))
        hk = hknew
        done && break
    end
    set_tols!(oracle, hk, delta)
    inject_gradient!(bw.gcoord, oracle, X, bw.x); cnt.ngrad += 1
    inject_linearization!(imap, oracle, X, Y, bw.x)
    riesz!(bw.dg, X, bw.gcoord)
    setmodel!(model, bw.x, bw.gcoord)
    reanchor!(psi, bw.x)
    hpost, okpost = hstat!(bw.step, bw.pgiter, psi, X, bw.x, bw.dg, opts.t0, cnt)
    oracle.hpost = hpost
    freeze_curvature!(model, psi.theta)                 # theta_k from the stationarity prox
    return max(hk, hpost), hok && okpost
end

# Trial step shared by both kernels: Cauchy point (`subsolver=:cauchy`) optionally
# improved by the SPG subsolver, then the TRUE reduction. Writes `bw.xtrial`;
# returns (pRed, worstpg, Jtrial). `worstpg` is the worst dual-SPG residual among
# the step's proxes (0 for exact proxes).
function _trial_step!(prob::RiskProblem, opts, psi, model, bw, sval, nval, delta, cnt)
    X = prob.X
    _, svalc, nvalc, _, worstpg = dcauchy!(bw.s, model, psi, X, bw.x, bw.dg, delta,
                                           opts.alpha0, sval, nval, bw.px, bw.dwa, bw.dwa1,
                                           cnt, opts.cauchy)
    @. bw.xtrial = bw.x + bw.s
    if opts.subsolver === :spg
        # Improve the Cauchy point with spectral proximal-gradient steps; the model
        # value only decreases, so the fraction-of-Cauchy-decrease still holds.
        svalc, nvalc, wpg = dspg!(bw.xtrial, model, psi, X, bw.x, bw.gcoord, delta,
                                  svalc, nvalc, bw.gmodc, bw.gmodp, bw.step, bw.pgiter, bw.dwa, bw.px,
                                  cnt, opts.spg)
        worstpg = max(worstpg, wpg)
    elseif opts.subsolver === :ncg
        # Improve the Cauchy point with nonlinear-CG steps (same monotone-model
        # guarantee). Scratch reuses the post-dcauchy! buffers + one extra (ncgaux).
        svalc, nvalc, wpg = dncg!(bw.xtrial, model, psi, X, bw.x, bw.gcoord, delta,
                                  svalc, nvalc, bw.s, bw.pgiter, bw.step, bw.px, bw.dwa1, bw.ncgaux,
                                  bw.dwa, bw.gmodc, bw.gmodp, cnt, opts.ncg)
        worstpg = max(worstpg, wpg)
    end
    pRed = (sval + nval) - (svalc + nvalc)              # model decrease m_k(x) - m_k(x+s)
    Jtrial = objective_value!(bw.Fbuf, prob, bw.xtrial); cnt.nfval += 1   # true J (exact F)
    return pRed, worstpg, Jtrial
end

# Accept/reject decision shared by both kernels: returns (reject::Bool, rho).
@inline function _accept_decision(rho, flag, worstpg, opts)
    # A step whose proxes merely missed the dual-SPG tol by a little is judged on its
    # true rho (force-rejecting these caused a death spiral on hard CVaR duals); a
    # grossly unreliable step (worstpg > step_prox_tol) is rejected (and, being
    # rejected, cannot grow the radius). Otherwise reject on nonpositive/NaN predicted
    # reduction (flag >= NPOSPREDPOS) or insufficient ratio; POSPREDNEG is accepted,
    # matching ROL.
    gross = worstpg > opts.step_prox_tol
    return gross || flag >= TR_NPOSPREDPOS || (rho < opts.eta1 && flag == TR_SUCCESS)
end

@inline function _final_status(converged, hok, stalled)
    return converged ? :converged :
           !hok      ? :prox_failure :   # unreliable stationarity prox dominates
           stalled   ? :stalled :
                       :maxiter
end

# Exact-evaluation kernel (no oracle): the trial reduction is the true ared.
function _solve_exact!(prob::RiskProblem{T}, opts, psi, model, bw, cnt) where {T}
    X = prob.X
    hk, hok = _refresh_exact!(prob, opts, psi, model, bw, cnt)
    sval = value(prob.f, bw.x); nval = value(psi, bw.x)
    cnt.nfval += 1; cnt.npsi += 1
    Jval = sval + nval                                  # exact true J(x_1)
    # Auto initial radius (delta0 <= 0): a data-driven scale that accounts for
    # NONSMOOTH stationarity, not just the smooth gradient. With f = 0 (or any
    # grad f(x_1) = 0) and an active risk/phi term, spnorm(X, dg) = 0 alone would give
    # a zero radius that false-stalls on iteration 1; the prox-gradient step (held in
    # bw.step after the initial hstat!) has norm t0*h_k > 0 whenever the nonsmooth term
    # moves the iterate. Both zero => x is already prox-stationary (handled by the
    # convergence/stall/status logic); the one(T) fallback applies ONLY when the prox
    # certified (hok), so it never masks a failed prox certification.
    delta = opts.delta0 > 0 ? opts.delta0 : zero(T)
    if opts.delta0 <= 0
        delta = max(spnorm(X, bw.dg), spnorm(X, bw.step))
        iszero(delta) && hok && (delta = one(T))
    end

    opts.verbose > 0 && _print_header()
    opts.verbose > 0 && _print_row(0, Jval, hk, delta, NaN, cnt)

    converged = false; stalled = false; iter = 0
    while iter < opts.maxit
        if hk <= opts.htol && hok                       # certified: h_k prox converged
            converged = true; break
        end
        if delta <= eps(T) * max(one(T), spnorm(X, bw.x))   # radius collapsed below iterate precision
            stalled = true; break
        end
        iter += 1
        pRed, worstpg, Jtrial = _trial_step!(prob, opts, psi, model, bw, sval, nval, delta, cnt)
        ared = Jval - Jtrial
        rho, flag = analyze_ratio_red(ared, pRed, Jval, opts.trsafe * eps(T))
        if _accept_decision(rho, flag, worstpg, opts)   # reject
            delta = opts.gamma1 * delta                 # paper rule (gamma1=gamma2)
        else                                            # accept
            copyto!(bw.x, bw.xtrial); Jval = Jtrial
            rho >= opts.eta2 && (delta = min(opts.gamma3 * delta, opts.deltamax))
            hk, hok = _refresh_exact!(prob, opts, psi, model, bw, cnt)
            sval = value(prob.f, bw.x); nval = value(psi, bw.x)
            cnt.nfval += 1; cnt.npsi += 1
        end
        opts.verbose > 0 && _print_row(iter, Jval, hk, delta, rho, cnt)
    end
    return TRResult{T,Vector{T}}(bw.x, Jval, hk, iter, converged, _final_status(converged, hok, stalled), cnt)
end

# Inexact-evaluation kernel (Conditions 1 & 2): the accept/reject test uses the
# computed reduction cred = ared + bounded error; the data is rebuilt whenever the
# radius changes (its Condition-1 tolerances depend on Δ).
function _solve_inexact!(prob::RiskProblem{T}, opts, psi, model, oracle, imap, bw, cnt) where {T}
    X = prob.X
    delta = opts.delta0                                 # validated > 0 for inexact
    hk, hok = _refresh_inexact!(prob, opts, psi, model, oracle, imap, bw, delta, delta, cnt)
    sval = value(prob.f, bw.x); nval = value(psi, bw.x)
    cnt.nfval += 1; cnt.npsi += 1
    Jval = objective_value!(bw.Fbuf, prob, bw.x); cnt.nfval += 1   # exact true J(x_1)

    opts.verbose > 0 && _print_header()
    opts.verbose > 0 && _print_row(0, Jval, hk, delta, NaN, cnt)

    converged = false; stalled = false; iter = 0
    while iter < opts.maxit
        if hk <= opts.htol && hok                       # certified: h_k prox converged
            converged = true; break
        end
        if delta <= eps(T) * max(one(T), spnorm(X, bw.x))   # radius collapsed below iterate precision
            stalled = true; break
        end
        iter += 1
        pRed, worstpg, Jtrial = _trial_step!(prob, opts, psi, model, bw, sval, nval, delta, cnt)
        ared = Jval - Jtrial                            # exact reference (uses prob.F, not the inexact map)
        cred, credbound = computed_reduction(oracle, ared, pRed, iter)
        rho, flag = analyze_ratio_red(cred, pRed, Jval, opts.trsafe * eps(T))
        push_record!(oracle, iter, hk, delta, pRed, ared, cred, credbound)
        if _accept_decision(rho, flag, worstpg, opts)   # reject
            delta = opts.gamma1 * delta
            # The inexact data tolerances depend on Δ, so rebuild at the same x.
            hk, hok = _refresh_inexact!(prob, opts, psi, model, oracle, imap, bw, hk, delta, cnt)
            sval = value(prob.f, bw.x); nval = value(psi, bw.x); cnt.nfval += 1; cnt.npsi += 1
        else                                            # accept
            copyto!(bw.x, bw.xtrial); Jval = Jtrial
            rho >= opts.eta2 && (delta = min(opts.gamma3 * delta, opts.deltamax))
            hk, hok = _refresh_inexact!(prob, opts, psi, model, oracle, imap, bw, hk, delta, cnt)
            sval = value(prob.f, bw.x); nval = value(psi, bw.x); cnt.nfval += 1; cnt.npsi += 1
        end
        # ζ_k -> 0 advances every outer iteration (the paper requires it over
        # iterations, not over accepted steps).
        oracle.zeta_k *= oracle.zfactor
        opts.verbose > 0 && _print_row(iter, Jval, hk, delta, rho, cnt)
    end
    return TRResult{T,Vector{T}}(bw.x, Jval, hk, iter, converged, _final_status(converged, hok, stalled), cnt)
end

"""
    solve(prob, x0; opts, oracle=nothing) -> TRResult

Run TR-Risk (paper Algorithm 1) from `x0`. `opts.subsolver` selects the trial step
(`:cauchy`, `:spg`, or `:ncg`); `opts.use_curvature` toggles the `(theta_k, F'')`
term in `B_k`. Passing an `InexactOracle` runs the inexact variant (Conditions 1 &
2), keeping `prob.F` as the exact reference for the objective.
"""
# Deterministic adjoint-consistency SMOKE CHECK (no Random). A correct VJP is the
# bare transpose of the JVP, so ⟨A v, θ⟩ = ⟨v, Aᵀθ⟩ (a transpose-mismatched VJP is
# O(1) off, relative). A few fixed `_dir` probe pairs — a guard, not a proof: an
# error exactly orthogonal to every probe would slip, hence "smoke check".
function _validate_F(F::VectorMap{T}, x0, tol) where {T}
    prepare!(F, x0)
    n = length(x0); m = rangedim(F)
    Jv = Vector{T}(undef, m); Jat = Vector{T}(undef, n)
    worst = zero(T)
    for (sv, st) in ((1, 2), (2, 0), (0, 1))     # three distinct deterministic probe pairs
        v = _dir(T, n, sv); theta = _dir(T, m, st)
        applyJacobian!(Jv, F, v, x0)
        applyAdjointJacobian!(Jat, F, theta, x0)
        a = dot(Jv, theta); b = dot(v, Jat)
        worst = max(worst, abs(a - b) / (abs(a) + one(T)))
    end
    worst > tol && throw(ArgumentError(
        "validate_F smoke check failed: relative adjoint residual $(worst) > $(tol). " *
        "applyAdjointJacobian! must return the bare transpose Aᵀθ, consistent with applyJacobian!."))
    return nothing
end

# Resolve the line-search reuse mode once: :on/:off force it; :auto follows the map's
# operator-actions-expensive trait (true only for matrix-free OperatorMap, and for an
# InexactMap wrapping one). Returns a Bool; the caller branches into a function
# barrier so PsiModel's reuse mode is a concrete (compile-time) singleton.
function _reuse_on(mode::Symbol, F)
    mode === :on && return true
    mode === :off && return false
    return operator_actions_expensive(typeof(F))     # :auto
end

function solve(prob::RiskProblem{T}, x0::AbstractVector{T};
               opts::TROptions{T}=TROptions{T}(),
               oracle::Union{Nothing,InexactOracle{T}}=nothing) where {T}
    n = length(x0); m = rangedim(prob.F)
    opts.validate_F && _validate_F(prob.F, x0, opts.validate_tol)
    if oracle !== nothing
        opts.delta0 <= 0 && throw(ArgumentError("inexact solve requires opts.delta0 > 0"))
        # Condition 2 requires 0 < eta < min(eta1, 1 - eta2).
        !(zero(T) < oracle.eta < min(opts.eta1, one(T) - opts.eta2)) &&
            throw(ArgumentError("InexactOracle eta must satisfy 0 < eta < min(eta1, 1-eta2)"))
    end
    # Resolve against prob.F; under inexactness the InexactMap wrapping prob.F forwards
    # the same trait, so the gate is identical. Then dispatch into a concrete kernel.
    return _reuse_on(opts.matrix_free_reuse, prob.F) ?
           _solve_resolved(prob, x0, opts, oracle, ReuseOn()) :
           _solve_resolved(prob, x0, opts, oracle, ReuseOff())
end

# Function barrier: with `ru` concrete, PsiModel's reuse mode is a compile-time type
# parameter, so the kernels specialize and the ReuseOff branch is elided.
function _solve_resolved(prob::RiskProblem{T}, x0, opts, oracle, ru) where {T}
    X = prob.X; Y = prob.Y
    n = length(x0); m = rangedim(prob.F)
    bw = _SolveBuffers(T, x0, m)
    cnt = Counters()
    if oracle === nothing
        psi = PsiModel(prob.phi, prob.A, prob.F, X, Y, n; opts=opts.dualspg, reuse=ru)
        model = opts.use_curvature ? QuadraticModel(prob.f, prob.F, X, Y, n, m) :
                                     QuadraticModel(prob.f, X, n)
        return _solve_exact!(prob, opts, psi, model, bw, cnt)
    else
        # The model linearization uses an InexactMap (perturbed F); prob.F is never
        # replaced, so objective_value! keeps the exact reference. Curvature's F''
        # delegates to the exact F (Condition 1 bounds g_k/b_k/A_k, not the Hessian).
        imap = InexactMap(prob.F, n)
        psi = PsiModel(prob.phi, prob.A, imap, X, Y, n; opts=opts.dualspg, reuse=ru)
        model = opts.use_curvature ? QuadraticModel(prob.f, imap, X, Y, n, m) :
                                     QuadraticModel(prob.f, X, n)
        return _solve_inexact!(prob, opts, psi, model, oracle, imap, bw, cnt)
    end
end

function _print_header()
    @printf("%6s %15s %12s %12s %10s %8s %8s %8s %8s\n",
            "iter", "J", "hk", "delta", "rho", "nfval", "ngrad", "nhess", "nprox")
end
function _print_row(iter, J, hk, delta, rho, cnt::Counters)
    @printf("%6d %15.8e %12.4e %12.4e %10.3f %8d %8d %8d %8d\n",
            iter, J, hk, delta, rho, cnt.nfval, cnt.ngrad, cnt.nhess, cnt.nprox)
end
