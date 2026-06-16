# Cauchy-only TR-Risk: the main loop (paper Algorithm 1) specialized to the
# Cauchy-point step. A valid instance of Algorithm 1 (which only requires
# sufficient model decrease); NCG/SPG improvers are Phase 2.
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
    delta0::T        # initial radius (<=0 => use ‖g_1‖)
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
    subsolver::Symbol  # :cauchy (Cauchy point only) or :spg (Cauchy + SPG improver)
    dualspg::DualSPGOptions{T}
    spg::SpgParams{T}
end
# Note `alpha0` defaults to `t0`: per the Phase-1 rule the Cauchy bidirectional
# search starts from the same prox-gradient parameter `t0` used for h_k.
function TROptions{T}(; delta0=10, deltamax=1e12, eta1=1e-4, eta2=0.5,
                      gamma1=0.25, gamma3=10, t0=1, alpha0=t0,
                      mu0=1e-2, interpf=0.1, extrapf=10, redlim=10, explim=10, qtol=1e-8,
                      trsafe=1e2, step_prox_tol=1e-6, htol=1e-8, maxit=1000, verbose=0,
                      subsolver::Symbol=:cauchy,
                      dualspg::DualSPGOptions{T}=DualSPGOptions{T}(),
                      spg::SpgParams{T}=SpgParams{T}()) where {T}
    subsolver in (:cauchy, :spg) || throw(ArgumentError("subsolver must be :cauchy or :spg"))
    cp = CauchyParams{T}(T(mu0), T(interpf), T(extrapf), Int(redlim), Int(explim), T(qtol))
    TROptions{T}(T(delta0), T(deltamax), T(eta1), T(eta2), T(gamma1), T(gamma3),
                 T(t0), T(alpha0), cp, T(trsafe), T(step_prox_tol), T(htol),
                 Int(maxit), Int(verbose), subsolver, dualspg, spg)
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

"""
    solve(prob, x0; opts) -> TRResult

Run Cauchy-only TR-Risk from `x0`.
"""
function solve(prob::RiskProblem{T}, x0::AbstractVector{T};
               opts::TROptions{T}=TROptions{T}(),
               oracle::Union{Nothing,InexactOracle{T}}=nothing) where {T}
    X = prob.X; Y = prob.Y
    n = length(x0)
    m = rangedim(prob.F)
    inexact = oracle !== nothing
    inexact && opts.delta0 <= 0 &&
        throw(ArgumentError("inexact solve requires opts.delta0 > 0"))
    # Condition 2 requires 0 < eta < min(eta1, 1 - eta2).
    inexact && !(zero(T) < oracle.eta < min(opts.eta1, one(T) - opts.eta2)) &&
        throw(ArgumentError("InexactOracle eta must satisfy 0 < eta < min(eta1, 1-eta2)"))
    # The model linearization uses an InexactMap (perturbed F) when inexact;
    # prob.F is never replaced, so objective_value! keeps the exact reference.
    imap = inexact ? InexactMap(prob.F, n) : nothing
    Fmodel = inexact ? imap : prob.F
    psi = PsiModel(prob.phi, prob.A, Fmodel, X, Y, n; opts=opts.dualspg)
    model = QuadraticModel(prob.f, X, n)
    cnt = Counters()

    x      = copyto!(zeros(T, n), x0)
    xtrial = zeros(T, n)
    gcoord = zeros(T, n)
    dg     = zeros(T, n)
    s      = zeros(T, n)     # Cauchy step
    step   = zeros(T, n)     # hstat step
    pgiter = zeros(T, n)
    px     = zeros(T, n)     # dcauchy trial iterate
    dwa    = zeros(T, n)
    dwa1   = zeros(T, n)
    gmodc  = zeros(T, n)     # SPG improver scratch
    gmodp  = zeros(T, n)
    Fbuf   = zeros(T, m)

    # Refresh the model data at `x` for radius `delta` and return (h_k, hok). Exact:
    # one pass. Inexact: warm-started fixed point that injects (g_k, A_k, b_k) at
    # tolerances kappa·min(h_k, Δ[^2]) and refines h_k, followed by a POSTCONDITION
    # injection consistent with the final h_k. The postcondition guarantees the data
    # used by the model/steps satisfies Condition 1 against the REPORTED h_k for any
    # kappa (the fixed point — a contraction only for kappa < 1 — just improves the
    # h_k estimate; correctness of the contract does not depend on its convergence).
    function refresh!(x, delta, hk0)
        if !inexact
            gradient!(gcoord, prob.f, x); cnt.ngrad += 1
            riesz!(dg, X, gcoord)
            setmodel!(model, x, gcoord)
            reanchor!(psi, x)
            return hstat!(step, pgiter, psi, X, x, dg, opts.t0, cnt)
        end
        hk = hk0; hok = false
        for _ in 1:10
            set_tols!(oracle, hk, delta)
            inject_gradient!(gcoord, oracle, X, x); cnt.ngrad += 1
            inject_linearization!(imap, oracle, X, Y, x)
            riesz!(dg, X, gcoord)
            setmodel!(model, x, gcoord)
            reanchor!(psi, x)                           # b_k = F(x) + e_b via imap
            hknew, hok = hstat!(step, pgiter, psi, X, x, dg, opts.t0, cnt)
            done = abs(hknew - hk) <= T(1e-3) * max(hknew, eps(T))
            hk = hknew
            done && break
        end
        # Postcondition: inject data consistent with the fixed-point h_k so the
        # model/steps satisfy Condition 1 against it; then recompute the metric for
        # THAT final model and return a conservative certified value max(hk, hpost)
        # (so the reported metric dominates the final model's true stationarity and
        # cannot falsely certify convergence). Condition 1 still holds because the
        # data error <= kappa·min(hk, Δ) <= kappa·min(max(hk,hpost), Δ).
        set_tols!(oracle, hk, delta)
        inject_gradient!(gcoord, oracle, X, x); cnt.ngrad += 1
        inject_linearization!(imap, oracle, X, Y, x)
        riesz!(dg, X, gcoord)
        setmodel!(model, x, gcoord)
        reanchor!(psi, x)
        hpost, okpost = hstat!(step, pgiter, psi, X, x, dg, opts.t0, cnt)
        oracle.hpost = hpost
        return max(hk, hpost), hok && okpost
    end

    # --- Initialize at x_1 ---
    delta = opts.delta0 > 0 ? opts.delta0 : zero(T)     # finalized below for exact
    hk, hok = refresh!(x, delta, opts.delta0 > 0 ? opts.delta0 : one(T))
    sval = value(prob.f, x)                             # f(x_1) (exact; cancels in pRed)
    nval = value(psi, x)                                # model nonsmooth at the anchor
    cnt.nfval += 1; cnt.npsi += 1
    Jval = inexact ? objective_value!(Fbuf, prob, x) : sval + nval   # exact true J(x_1)
    inexact && (cnt.nfval += 1)
    !inexact && opts.delta0 <= 0 && (delta = spnorm(X, dg))

    opts.verbose > 0 && _print_header()
    opts.verbose > 0 && _print_row(0, Jval, hk, delta, NaN, cnt)

    converged = false
    stalled = false
    iter = 0
    while iter < opts.maxit
        # Certified convergence requires the h_k-defining prox to have converged.
        if hk <= opts.htol && hok
            converged = true
            break
        end
        # Stall guard: if the radius has collapsed below the precision of the
        # iterate, no representable step can make further progress (this happens
        # when the achievable stationarity floor is above htol, e.g. with large
        # inexactness). Stop instead of spinning to maxit.
        if delta <= eps(T) * max(one(T), spnorm(X, x))
            stalled = true
            break
        end
        iter += 1

        # Cauchy step (model anchored/frozen at current x). `worstpg` is the worst
        # dual-SPG residual among the step's proxes (0 for exact proxes).
        _, svalc, nvalc, _, worstpg = dcauchy!(s, model, psi, X, x, dg, delta,
                                               opts.alpha0, sval, nval, px, dwa, dwa1,
                                               cnt, opts.cauchy)
        @. xtrial = x + s
        if opts.subsolver === :spg
            # Improve the Cauchy point with spectral proximal-gradient steps; the
            # model value only decreases, so the fraction-of-Cauchy-decrease holds.
            svalc, nvalc, wpg = dspg!(xtrial, model, psi, X, x, gcoord, delta,
                                      svalc, nvalc, gmodc, gmodp, step, pgiter, dwa, px,
                                      cnt, opts.spg)
            worstpg = max(worstpg, wpg)
        end
        pRed = (sval + nval) - (svalc + nvalc)          # model decrease m_k(x) - m_k(x+s)

        # Exact reference reduction (the truth uses prob.F, never the inexact map).
        Jtrial = objective_value!(Fbuf, prob, xtrial); cnt.nfval += 1
        ared = Jval - Jtrial
        # The accept/reject decision uses the COMPUTED reduction: exact ared in the
        # exact case; cred = ared + bounded error (Condition 2) in the inexact case.
        if inexact
            cred, credbound = computed_reduction(oracle, ared, pRed, iter)
            rho, flag = analyze_ratio_red(cred, pRed, Jval, opts.trsafe * eps(T))
            push_record!(oracle, iter, hk, delta, pRed, ared, cred, credbound)
        else
            cred = ared
            rho, flag = analyze_ratio_red(ared, pRed, Jval, opts.trsafe * eps(T))
        end

        # Graded prox-reliability policy:
        #  * A step whose proxes merely missed the dual-SPG tol by a little
        #    (worstpg <= step_prox_tol) is judged on its true rho — pRed/cred are
        #    evaluated at the point actually reached. (Force-rejecting these caused
        #    a death spiral on hard CVaR duals that converge just shy of tol.)
        #  * A grossly unreliable step (worstpg > step_prox_tol) is rejected and,
        #    being rejected, cannot grow the radius.
        # Otherwise reject on nonpositive/NaN predicted reduction (flags >=
        # NPOSPREDPOS) or insufficient ratio; POSPREDNEG is accepted, matching ROL.
        # Certified convergence still requires the h_k-defining prox to converge.
        gross = worstpg > opts.step_prox_tol
        reject = gross || flag >= TR_NPOSPREDPOS || (rho < opts.eta1 && flag == TR_SUCCESS)
        if reject
            delta = opts.gamma1 * delta                 # paper rule (gamma1=gamma2);
                                                        # ROL uses gamma1*min(snorm,delta)
            if inexact
                # The inexact data tolerances depend on Δ (Condition 1), so the data
                # must be rebuilt at the same x for the shrunken radius.
                hk, hok = refresh!(x, delta, hk)
                sval = value(prob.f, x); nval = value(psi, x)
                cnt.nfval += 1; cnt.npsi += 1
            end
        else                                            # accept
            copyto!(x, xtrial)
            Jval = Jtrial
            if rho >= opts.eta2
                delta = min(opts.gamma3 * delta, opts.deltamax)
            end
            hk, hok = refresh!(x, delta, hk)            # re-anchor + inexact data at new x (warm)
            sval = value(prob.f, x)
            nval = value(psi, x)
            cnt.nfval += 1; cnt.npsi += 1
        end
        # Forcing sequence ζ_k -> 0 advances every outer iteration (the paper requires
        # ζ_k -> 0 over iterations, not over accepted steps).
        inexact && (oracle.zeta_k *= oracle.zfactor)

        opts.verbose > 0 && _print_row(iter, Jval, hk, delta, rho, cnt)
    end

    status = converged ? :converged :
             !hok      ? :prox_failure :   # unreliable stationarity prox dominates
             stalled   ? :stalled :
                         :maxiter
    return TRResult{T,Vector{T}}(x, Jval, hk, iter, converged, status, cnt)
end

function _print_header()
    @printf("%6s %15s %12s %12s %10s %8s %8s %8s %8s\n",
            "iter", "J", "hk", "delta", "rho", "nfval", "ngrad", "nhess", "nprox")
end
function _print_row(iter, J, hk, delta, rho, cnt::Counters)
    @printf("%6d %15.8e %12.4e %12.4e %10.3f %8d %8d %8d %8d\n",
            iter, J, hk, delta, rho, cnt.nfval, cnt.ngrad, cnt.nhess, cnt.nprox)
end
