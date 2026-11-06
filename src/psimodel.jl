# Part of ROLOpt.jl (MIT; see LICENSE). Portions are ported from Trilinos ROL,
# Copyright (2014) Sandia Corporation, under its BSD-style license (see LICENSE-ROL).

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

# Reuse-mode tags (Phase 2). `solve` resolves opts.matrix_free_reuse + the map's
# `operator_actions_expensive` trait into one of these ONCE and stores it as a
# PsiModel type parameter, so the hot kernels (prox!, dbls!/dncg!) dispatch on it at
# compile time: ReuseOff elides the line-search operator-reuse branch entirely, so
# the explicit/cached paths stay byte-for-byte (and zero-alloc). ReuseOn precomputes
# A_k applies along the fixed line-search direction and AXPYs per trial (matrix-free).
struct ReuseOn end
struct ReuseOff end

"""
Status returned by the dual spectral projected gradient (Algorithm 2). `isbits`,
so returning it from `prox!` allocates nothing. The optimal dual is NOT carried
here — `prox!` leaves it in `psi.theta`; a caller needing a snapshot across a
later `prox!` must `copy(psi.theta)` first.
"""
struct DualSPGResult{T}
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
    memory >= 1 || throw(ArgumentError("DualSPGOptions requires memory >= 1"))
    DualSPGOptions{T}(T(lam0), T(gmin), T(gmax), T(sig1), T(sig2), T(alpha),
                      T(beta), Int(memory), T(tol), Int(maxit), Int(lsmax))
end

"""Preallocated scratch for `PsiModel` (X-space length n, Y-space length m)."""
struct PsiWorkspace{T,VX<:AbstractVector{T},VY<:AbstractVector{T}}
    w::VX; adjp::VX; pcur::VX; ptrial::VX; dz::VX; xtmp::VX; teff::VX  # X primal
    adj::VX                                                           # X coord (Aᵀ(M_Y theta))
    qadj::VX                                                          # X scratch for the QuadraticMap adjoint gemv
    ath::VX; asx::VX                                                  # X primal: A_k^*theta, A_k^*s (dual-SPG reuse)
    ell::VY                                                           # Y (value() scratch)
    mu::VY; mutrial::VY; s::VY; sprev::VY; muprev::VY                 # Y
    tmptheta::VY; thetatrial::VY; ytmp::VY                            # Y (ytmp = M_Y theta)
    ly::VY; ay::VY; ltr::VY                                           # Y primal: ell_k(y), A_k·s, ell trial (dbls! reuse)
    dq::Vector{T}                                                     # nonmonotone d-memory ring (length = memory)
    dqn::Base.RefValue{Int}                                          # valid ring entries, 0..memory
    dqpos::Base.RefValue{Int}                                        # last write slot, 0 then 1..memory cycling
end
function PsiWorkspace(::Type{T}, n::Integer, m::Integer, memory::Integer) where {T}
    vx() = zeros(T, n)
    vy() = zeros(T, m)
    PsiWorkspace{T,Vector{T},Vector{T}}(vx(), vx(), vx(), vx(), vx(), vx(), vx(), vx(), vx(),
        vx(), vx(),
        vy(), vy(), vy(), vy(), vy(), vy(), vy(), vy(), vy(),
        vy(), vy(), vy(),
        zeros(T, Int(memory)), Ref(0), Ref(0))
end

# Fixed-capacity nonmonotone d-memory ring (no push!/popfirst!/view allocations).
@inline _dq_reset!(ws::PsiWorkspace) = (ws.dqn[] = 0; ws.dqpos[] = 0; ws)
@inline function _dq_push!(ws::PsiWorkspace, d)
    m = length(ws.dq)
    pos = ws.dqpos[] % m + 1                  # 1..m cycling
    ws.dqpos[] = pos
    @inbounds ws.dq[pos] = d
    ws.dqn[] = min(ws.dqn[] + 1, m)
    return ws
end
# min over the active entries — valid slots are 1..dqn[] (sequential before wrap;
# all m once dqn[] == m), so a plain min loop over 1:dqn[] is exact and alloc-free.
@inline function _dq_min(ws::PsiWorkspace{T}) where {T}
    @inbounds r = ws.dq[1]
    @inbounds for k in 2:ws.dqn[]
        ws.dq[k] < r && (r = ws.dq[k])
    end
    return r
end

"""
    PsiModel(phi, A, F, X, Y; n, m, opts)

The re-anchorable nonsmooth model term `psi_k`. Call `reanchor!(psi, x_k)` once
per outer iteration to set the linearization `(A_k, b_k, x_k)`.
"""
struct PsiModel{T,P,R,FM,SX,SY,VX,VY,RU}
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
    reuse::RU      # ReuseOn / ReuseOff (compile-time line-search reuse gate)
end
function PsiModel(phi::ProxObjective{T}, A::RiskEnvelope{T}, F::VectorMap{T},
                  X::HilbertSpace{T}, Y::HilbertSpace{T}, n::Integer;
                  opts::DualSPGOptions{T}=DualSPGOptions{T}(), reuse=ReuseOff()) where {T}
    m = rangedim(F)
    xk = zeros(T, n); bk = zeros(T, m); theta = zeros(T, m)
    initdual!(theta, A)
    ws = PsiWorkspace(T, n, m, opts.memory)
    PsiModel{T,typeof(phi),typeof(A),typeof(F),typeof(X),typeof(Y),Vector{T},Vector{T},typeof(reuse)}(
        phi, A, F, X, Y, xk, bk, theta, opts, ws, reuse)
end

"""Re-anchor the linearization at `x_k`: sets `x_k`, runs the `prepare!` hook
(PDE factorization / cached-Jacobian assembly), then `b_k = F(x_k)`."""
function reanchor!(psi::PsiModel, xk)
    copyto!(psi.xk, xk)
    # Both the prepare! hook and b_k use the SAME frozen anchor psi.xk — the cache the
    # map builds in prepare! is exactly what every subsequent apply at psi.xk reuses.
    prepare!(psi.F, psi.xk)      # build the frozen linearization at x_k (no-op for plain maps)
    value!(psi.bk, psi.F, psi.xk)
    # Moving-sample maps re-index their output set each anchor, so a dual theta warm-
    # started from the previous anchor refers to stale slots; re-seed it from the fresh
    # anchor values b_k via warmdual! (for the simplex/max envelope this is the argmax
    # vertex — a far better start than uniform). Fixed-index maps keep the warm start.
    moving_sample(psi.F) && warmdual!(psi.theta, psi.A, psi.bk)
    return psi
end

# psi_k(x) = suppA(ell_k(x)) + phi(x),  ell_k(x) = A_k(x - x_k) + b_k.
function value(psi::PsiModel, x)
    ws = psi.ws
    @. ws.xtmp = x - psi.xk
    applyJacobian!(ws.ell, psi.F, ws.xtmp, psi.xk)
    ws.ell .+= psi.bk
    return suppA(psi.A, psi.Y, ws.ell) + value(psi.phi, x)
end

# A_k^* theta = riesz_X(Aᵀ (M_Y theta)) (primal X), written into ws.adjp and returned.
# (A_k v, theta)_Y = (v, A_k^* theta)_X under the X/Y metrics.
function _compute_adj!(psi::PsiModel, theta)
    ws = psi.ws
    metric!(ws.ytmp, psi.Y, theta)                       # M_Y theta (coord Y)
    applyAdjointJacobian!(ws.adj, psi.F, ws.ytmp, psi.xk, ws.qadj) # Aᵀ (M_Y theta) (coord X)
    riesz!(ws.adjp, psi.X, ws.adj)                        # A_k^* theta (primal X)
    return ws.adjp
end

# d(theta) given the precomputed adjoint adjp = A_k^* theta: writes p(theta) into pbuf
# and ell_k(p) (= grad d) into mubuf, returns d. The forward JVP A_k(p - x_k) is NOT
# affine in a line-search step (the prox bends), so it is always recomputed here.
function _dual_d_from_adj!(psi::PsiModel{T}, z, t, theta, adjp, pbuf, mubuf) where {T}
    ws = psi.ws
    @. ws.w = z - t * adjp
    proxX!(pbuf, psi.phi, psi.X, ws.w, t, ws.teff)       # p = prox^X_{t phi}(...)
    @. ws.xtmp = pbuf - psi.xk
    applyJacobian!(mubuf, psi.F, ws.xtmp, psi.xk)         # A_k(p - x_k)
    mubuf .+= psi.bk                                      # + b_k = ell_k(p) = grad d
    @. ws.dz = pbuf - z
    quad = inner(psi.X, ws.dz, ws.dz) / (2t)
    return quad + value(psi.phi, pbuf) + inner(psi.Y, theta, mubuf)
end

# Full d(theta): adjoint apply then the rest. Used by hstat! and the initial prox!
# evaluation; the reuse-aware line search uses the two pieces directly.
function _dual_d!(psi::PsiModel{T}, z, t, theta, pbuf, mubuf) where {T}
    adjp = _compute_adj!(psi, theta)
    return _dual_d_from_adj!(psi, z, t, theta, adjp, pbuf, mubuf)
end

# Line-search trial adjoint A_k^*(theta + lam*s) (primal X), into ws.adjp. ReuseOff
# recomputes it (one adjoint apply per trial); ReuseOn AXPYs the precomputed
# ws.ath = A_k^*theta and ws.asx = A_k^*s (no adjoint apply — the adjoint is linear).
_trial_adj!(psi::PsiModel, ::ReuseOff, thetatrial, lam) = _compute_adj!(psi, thetatrial)
function _trial_adj!(psi::PsiModel, ::ReuseOn, thetatrial, lam)
    ws = psi.ws
    @. ws.adjp = ws.ath + lam * ws.asx
    return ws.adjp
end

# prox_{t psi}(z) via Algorithm 2.  Writes the primal prox into `out` and returns
# an isbits `DualSPGResult` (status only). The optimal dual is left in `psi.theta`
# (warm-start cache); snapshot it with `copy(psi.theta)` before the next `prox!`.
function prox!(out, psi::PsiModel{T}, z, t) where {T}
    ws = psi.ws; o = psi.opts; A = psi.A; Y = psi.Y
    theta = psi.theta
    project!(theta, A, Y, theta)                          # feasible warm start
    dcur = _dual_d!(psi, z, t, theta, ws.pcur, ws.mu)     # mu = grad d(theta)
    psi.reuse isa ReuseOn && copyto!(ws.ath, ws.adjp)     # cache A_k^*theta for the line search
    _dq_reset!(ws); _dq_push!(ws, dcur)
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
        project!(ws.s, A, Y, ws.tmptheta)
        @. ws.s -= theta
        slope = inner(Y, ws.mu, ws.s)                    # (mu, s)_Y >= 0
        pgnorm = spnorm(Y, ws.s) / gamma
        if pgnorm <= o.tol
            converged = true; break
        end
        # ReuseOn: one adjoint apply for the fixed direction s; trials then AXPY it.
        psi.reuse isa ReuseOn && copyto!(ws.asx, _compute_adj!(psi, ws.s))
        # nonmonotone line search for lambda in (0, 1]
        dref = _dq_min(ws)
        lam = o.lam0
        @. ws.thetatrial = theta + lam * ws.s
        dtrial = _dual_d_from_adj!(psi, z, t, ws.thetatrial,
                                   _trial_adj!(psi, psi.reuse, ws.thetatrial, lam), ws.ptrial, ws.mutrial)
        ls = 0
        while dtrial < dref + o.alpha * lam * slope && ls < o.lsmax
            denomls = dtrial - dcur - lam * slope
            delta = denomls != zero(T) ? -(lam * lam * slope) / (2 * denomls) : o.beta * lam
            lam = (o.sig1 * lam <= delta <= o.sig2 * lam) ? delta : o.beta * lam
            @. ws.thetatrial = theta + lam * ws.s
            dtrial = _dual_d_from_adj!(psi, z, t, ws.thetatrial,
                                       _trial_adj!(psi, psi.reuse, ws.thetatrial, lam), ws.ptrial, ws.mutrial)
            ls += 1
        end
        # accept: stash previous gradient/step, advance theta and cached p/mu
        @. ws.muprev = ws.mu
        @. ws.sprev = ws.s
        # A_k^*theta_new = A_k^*theta + lam*A_k^*s (free — no adjoint apply).
        psi.reuse isa ReuseOn && (@. ws.ath = ws.ath + lam * ws.asx)
        lamprev = lam
        have_prev = true
        copyto!(theta, ws.thetatrial)
        copyto!(ws.pcur, ws.ptrial)
        copyto!(ws.mu, ws.mutrial)
        dcur = dtrial
        _dq_push!(ws, dcur)
    end
    copyto!(out, ws.pcur)
    return DualSPGResult{T}(nit, pgnorm, converged)
end
