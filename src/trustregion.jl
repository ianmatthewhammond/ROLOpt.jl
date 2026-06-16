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
function analyze_ratio(fold::T, ftrial::T, pRed::T, epsi::T) where {T}
    eps = epsi * max(one(T), abs(fold))
    aRed = fold - ftrial
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
    dualspg::DualSPGOptions{T}
end
# Note `alpha0` defaults to `t0`: per the Phase-1 rule the Cauchy bidirectional
# search starts from the same prox-gradient parameter `t0` used for h_k.
function TROptions{T}(; delta0=10, deltamax=1e12, eta1=1e-4, eta2=0.5,
                      gamma1=0.25, gamma3=10, t0=1, alpha0=t0,
                      mu0=1e-2, interpf=0.1, extrapf=10, redlim=10, explim=10, qtol=1e-8,
                      trsafe=1e2, step_prox_tol=1e-6, htol=1e-8, maxit=1000, verbose=0,
                      dualspg::DualSPGOptions{T}=DualSPGOptions{T}()) where {T}
    cp = CauchyParams{T}(T(mu0), T(interpf), T(extrapf), Int(redlim), Int(explim), T(qtol))
    TROptions{T}(T(delta0), T(deltamax), T(eta1), T(eta2), T(gamma1), T(gamma3),
                 T(t0), T(alpha0), cp, T(trsafe), T(step_prox_tol), T(htol),
                 Int(maxit), Int(verbose), dualspg)
end

"""
Result of `solve`. `converged` is true only when `h_k <= htol` AND the prox that
produced that `h_k` itself converged. `status` is one of `:converged`,
`:prox_failure` (the stationarity prox hit its iteration cap, so `h_k` is
unreliable), or `:maxiter`.
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
               opts::TROptions{T}=TROptions{T}()) where {T}
    X = prob.X; Y = prob.Y
    n = length(x0)
    m = rangedim(prob.F)
    psi = PsiModel(prob.phi, prob.A, prob.F, X, Y, n; opts=opts.dualspg)
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
    Fbuf   = zeros(T, m)

    # --- Initialize at x_1 ---
    reanchor!(psi, x)                                   # A_1, b_1 = F(x_1)
    sval = value(prob.f, x)                             # f(x_1)
    nval = value(psi, x)                                # psi_1(x_1) = suppA(F(x_1)) + phi(x_1)
    cnt.nfval += 1; cnt.npsi += 1
    Jval = sval + nval                                  # true J(x_1)
    gradient!(gcoord, prob.f, x); cnt.ngrad += 1
    riesz!(dg, X, gcoord)
    setmodel!(model, x, gcoord)
    hk, hok = hstat!(step, pgiter, psi, X, x, dg, opts.t0, cnt)
    delta = opts.delta0 > 0 ? opts.delta0 : spnorm(X, dg)

    opts.verbose > 0 && _print_header()
    opts.verbose > 0 && _print_row(0, Jval, hk, delta, NaN, cnt)

    converged = false
    iter = 0
    while iter < opts.maxit
        # Certified convergence requires the h_k-defining prox to have converged.
        if hk <= opts.htol && hok
            converged = true
            break
        end
        iter += 1

        # Cauchy step (model anchored/frozen at current x). `worstpg` is the worst
        # dual-SPG residual among the step's proxes (0 for exact proxes).
        _, svalc, nvalc, _, worstpg = dcauchy!(s, model, psi, X, x, dg, delta,
                                               opts.alpha0, sval, nval, px, dwa, dwa1,
                                               cnt, opts.cauchy)
        @. xtrial = x + s
        pRed = (sval + nval) - (svalc + nvalc)          # model decrease m_k(x) - m_k(x+s)

        # True actual reduction, with ROL's roundoff-safeguarded, flagged ratio test.
        Jtrial = objective_value!(Fbuf, prob, xtrial); cnt.nfval += 1
        rho, flag = analyze_ratio(Jval, Jtrial, pRed, opts.trsafe * eps(T))

        # Graded prox-reliability policy:
        #  * A step whose proxes merely missed the dual-SPG tol by a little
        #    (worstpg <= step_prox_tol) is judged on its true rho — pRed/aRed are
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
        else                                            # accept
            copyto!(x, xtrial)
            Jval = Jtrial
            reanchor!(psi, x)                           # re-anchor at new iterate
            sval = value(prob.f, x)
            nval = value(psi, x)
            cnt.nfval += 1; cnt.npsi += 1
            gradient!(gcoord, prob.f, x); cnt.ngrad += 1
            riesz!(dg, X, gcoord)
            setmodel!(model, x, gcoord)
            hk, hok = hstat!(step, pgiter, psi, X, x, dg, opts.t0, cnt)
            if rho >= opts.eta2
                delta = min(opts.gamma3 * delta, opts.deltamax)
            end
        end

        opts.verbose > 0 && _print_row(iter, Jval, hk, delta, rho, cnt)
    end

    status = converged ? :converged : (hok ? :maxiter : :prox_failure)
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
