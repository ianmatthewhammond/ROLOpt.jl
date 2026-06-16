# psi_k model objective and Algorithm 2 (the paper's net-new pieces).
#
#   psi_k(x) = suppA( A_k (x - x_k) + b_k ) + phi(x),     A_k ~ F'(x_k), b_k ~ F(x_k)
#
# This is the nonsmooth term consumed by the trust-region subsolvers in place of a
# plain phi. Its prox is NOT analytic; we compute it via a nonmonotone spectral
# projected gradient on the dual problem over theta in A (paper Algorithm 2).
#
# Dual problem (paper eq. dualprox): for prox argument z and parameter t = r,
#   p(theta) = prox_{t phi}( z - t * riesz_X(A_k^* theta) )
#   grad d(theta) = A_k( p(theta) - x_k ) + b_k =: ell_k(p(theta))   (in Y)
#   d(theta) = 1/(2t) ||p - z||_X^2 + phi(p) + (theta, ell_k(p))_Y   (concave)
# We MAXIMIZE d over A; the prox of psi_k is p(theta_bar).
#
# Primal/dual bookkeeping (matters for weighted spaces; trivial for Euclidean):
#   * gradient!/applyAdjointJacobian! return COORDINATE (dual) vectors -> map to
#     primal with riesz! before combining with primal vectors (the prox argument).
#   * dual variable theta and grad d live in Y (primal); all dual-side inner
#     products / norms use the Y HilbertSpace.
#
# Blueprint: ROL_TypeP_SpectralGradientAlgorithm_Def.hpp (nonmonotone BB SPG with
# the safeguarded quadratic-interpolation line search) + paper Algorithm 2.

"""Diagnostics returned by the dual spectral projected gradient (Algorithm 2)."""
struct DualSPGResult{T,VY<:AbstractVector{T}}
    theta::VY        # dual solution
    iters::Int       # outer dual-SPG iterations
    pgnorm::T        # final projected-gradient norm ||s||_Y / gamma
    converged::Bool  # false if the iteration cap was hit
end

"""Algorithm 2 parameters (defaults from the paper's Numerics section)."""
struct DualSPGOptions{T}
    lam0::T; gmin::T; gmax::T; sig1::T; sig2::T; alpha::T; beta::T
    memory::Int; tol::T; maxit::Int; lsmax::Int
end
function DualSPGOptions{T}(; lam0=1, gmin=1e-6, gmax=1e6, sig1=0.1, sig2=0.9,
                           alpha=1e-4, beta=0.5, memory=1, tol=1e-10,
                           maxit=1000, lsmax=50) where {T}
    DualSPGOptions{T}(T(lam0), T(gmin), T(gmax), T(sig1), T(sig2), T(alpha),
                      T(beta), Int(memory), T(tol), Int(maxit), Int(lsmax))
end

"""Preallocated scratch for `PsiModel` (X-space length n, Y-space length m)."""
struct PsiWorkspace{T,VX<:AbstractVector{T},VY<:AbstractVector{T}}
    w::VX; adjp::VX; pcur::VX; ptrial::VX; dz::VX; xtmp::VX  # X primal
    adj::VX                                                  # X coord (A_k^* theta)
    ell::VY                                                  # Y (value() scratch)
    mu::VY; mutrial::VY; s::VY; sprev::VY; muprev::VY        # Y
    tmptheta::VY; thetatrial::VY                             # Y
    dq::Vector{T}                                            # nonmonotone d-memory
end
function PsiWorkspace(::Type{T}, n::Integer, m::Integer) where {T}
    vx() = zeros(T, n)
    vy() = zeros(T, m)
    PsiWorkspace{T,Vector{T},Vector{T}}(vx(), vx(), vx(), vx(), vx(), vx(), vx(),
        vy(), vy(), vy(), vy(), vy(), vy(), vy(), vy(), T[])
end

"""
    PsiModel(phi, A, F, X, Y; n, m, opts)

The re-anchorable nonsmooth model term `psi_k`. Call `reanchor!(psi, x_k)` once
per outer iteration to set the linearization `(A_k, b_k, x_k)`.
"""
struct PsiModel{T,P,R,FM,SX,SY,VX,VY}
    phi::P
    A::R
    F::FM
    X::SX
    Y::SY
    xk::VX
    bk::VY
    theta::VY
    opts::DualSPGOptions{T}
    ws::PsiWorkspace{T,VX,VY}
end
function PsiModel(phi::ProxObjective{T}, A::RiskEnvelope{T}, F::VectorMap{T},
                  X::HilbertSpace{T}, Y::HilbertSpace{T}, n::Integer;
                  opts::DualSPGOptions{T}=DualSPGOptions{T}()) where {T}
    m = rangedim(F)
    xk = zeros(T, n); bk = zeros(T, m); theta = zeros(T, m)
    initdual!(theta, A)
    ws = PsiWorkspace(T, n, m)
    PsiModel{T,typeof(phi),typeof(A),typeof(F),typeof(X),typeof(Y),Vector{T},Vector{T}}(
        phi, A, F, X, Y, xk, bk, theta, opts, ws)
end

"""Re-anchor the linearization at `x_k`: sets `x_k` and `b_k = F(x_k)`."""
function reanchor!(psi::PsiModel, xk)
    copyto!(psi.xk, xk)
    value!(psi.bk, psi.F, xk)
    return psi
end

# psi_k(x) = suppA(ell_k(x)) + phi(x),  ell_k(x) = A_k(x - x_k) + b_k.
function value(psi::PsiModel, x)
    ws = psi.ws
    @. ws.xtmp = x - psi.xk
    applyJacobian!(ws.ell, psi.F, ws.xtmp, psi.xk)
    ws.ell .+= psi.bk
    return suppA(psi.A, ws.ell) + value(psi.phi, x)
end

# d(theta): writes p(theta) into pbuf and ell_k(p) (= grad d) into mubuf, returns d.
function _dual_d!(psi::PsiModel{T}, z, t, theta, pbuf, mubuf) where {T}
    ws = psi.ws
    applyAdjointJacobian!(ws.adj, psi.F, theta, psi.xk)   # A_k^* theta (coord X)
    riesz!(ws.adjp, psi.X, ws.adj)                        # -> primal X
    @. ws.w = z - t * ws.adjp
    prox!(pbuf, psi.phi, ws.w, t)                         # p = prox_{t phi}(...)
    @. ws.xtmp = pbuf - psi.xk
    applyJacobian!(mubuf, psi.F, ws.xtmp, psi.xk)         # A_k(p - x_k)
    mubuf .+= psi.bk                                      # + b_k = ell_k(p) = grad d
    @. ws.dz = pbuf - z
    quad = inner(psi.X, ws.dz, ws.dz) / (2t)
    return quad + value(psi.phi, pbuf) + inner(psi.Y, theta, mubuf)
end

# prox_{t psi}(z) via Algorithm 2.  Writes the primal prox into `out`.
function prox!(out, psi::PsiModel{T}, z, t) where {T}
    ws = psi.ws; o = psi.opts; A = psi.A; Y = psi.Y
    theta = psi.theta
    project!(theta, A, theta)                             # feasible warm start
    dcur = _dual_d!(psi, z, t, theta, ws.pcur, ws.mu)     # mu = grad d(theta)
    empty!(ws.dq); push!(ws.dq, dcur)
    gamma = clamp(one(T) / max(spnorm(Y, ws.mu), eps(T)), o.gmin, o.gmax)
    lamprev = o.lam0
    pgnorm = T(Inf); converged = false; nit = 0
    have_prev = false
    for n in 1:o.maxit
        nit = n
        if have_prev
            # BB step: gamma = lam_prev ||s_prev||^2 / (mu_prev - mu, s_prev)
            @. ws.muprev = ws.muprev - ws.mu             # muprev now holds (mu_prev - mu)
            denom = inner(Y, ws.muprev, ws.sprev)
            num = lamprev * inner(Y, ws.sprev, ws.sprev)
            gamma = denom > zero(T) ? clamp(num / denom, o.gmin, o.gmax) : o.gmax
        end
        # projected ascent step s = projA(theta + gamma*mu) - theta
        @. ws.tmptheta = theta + gamma * ws.mu
        project!(ws.s, A, ws.tmptheta)
        @. ws.s -= theta
        slope = inner(Y, ws.mu, ws.s)                    # (mu, s)_Y >= 0
        pgnorm = spnorm(Y, ws.s) / gamma
        if pgnorm <= o.tol
            converged = true; break
        end
        # nonmonotone line search for lambda in (0, 1]
        dref = minimum(ws.dq)
        lam = o.lam0
        @. ws.thetatrial = theta + lam * ws.s
        dtrial = _dual_d!(psi, z, t, ws.thetatrial, ws.ptrial, ws.mutrial)
        ls = 0
        while dtrial < dref + o.alpha * lam * slope && ls < o.lsmax
            denomls = dtrial - dcur - lam * slope
            delta = denomls != zero(T) ? -(lam * lam * slope) / (2 * denomls) : o.beta * lam
            lam = (o.sig1 * lam <= delta <= o.sig2 * lam) ? delta : o.beta * lam
            @. ws.thetatrial = theta + lam * ws.s
            dtrial = _dual_d!(psi, z, t, ws.thetatrial, ws.ptrial, ws.mutrial)
            ls += 1
        end
        # accept: stash previous gradient/step, advance theta and cached p/mu
        @. ws.muprev = ws.mu
        @. ws.sprev = ws.s
        lamprev = lam
        have_prev = true
        copyto!(theta, ws.thetatrial)
        copyto!(ws.pcur, ws.ptrial)
        copyto!(ws.mu, ws.mutrial)
        dcur = dtrial
        push!(ws.dq, dcur)
        while length(ws.dq) > o.memory
            popfirst!(ws.dq)
        end
    end
    copyto!(out, ws.pcur)
    return DualSPGResult{T,Vector{T}}(copyto!(similar(theta), theta), nit, pgnorm, converged)
end
