# Part of ROLOpt.jl (MIT; see LICENSE). Portions are ported from Trilinos ROL,
# Copyright (2014) Sandia Corporation, under its BSD-style license (see LICENSE-ROL).

# Inexactness (paper Conditions 1 & 2), wired explicitly and audit-friendly.
#
# Condition 1 (model data accuracy at x_k):
#   ‖g_k - ∇f(x_k)‖_X            <= kgrad · min(h_k, Δ_k)
#   ‖b_k - F(x_k)‖_Y             <= kval  · min(h_k, Δ_k^2)     (note Δ^2)
#   ‖A_k - F'(x_k)‖_{L(X,Y)}     <= kjac  · min(h_k, Δ_k)
# Condition 2 (computed reduction):
#   |ared - cred|                <= kobj · [η · min(pred, ζ_k)]^ζ,   ζ_k -> 0.
#
# Design: `prob.F` is NEVER replaced — `objective_value!` keeps the EXACT truth
# (the reference `ared`). The inexact linearization is supplied to the model's
# PsiModel through an `InexactMap` that wraps the exact `F` and adds a controlled
# perturbation `(E, e_b)`: A_k = F'(x_k) + E, b_k = F(x_k) + e_b. The oracle injects
# deterministic, norm-controlled perturbations and records the realized errors so
# the contract can be checked directly (not inferred from noise).

# ---------------------------------------------------------------------------
# Metric operator norm ‖E‖_{L(X,Y)} for a diagonal-metric X (cols) and Y (rows):
# the spectral norm of diag(√wY)·E·diag(1/√wX).
# ---------------------------------------------------------------------------
function opnorm_metric(E::AbstractMatrix{T}, X::HilbertSpace, Y::HilbertSpace) where {T}
    wX = proxweights(X); wY = proxweights(Y)
    M = copy(E)
    wY !== nothing && lmul!(Diagonal(sqrt.(wY)), M)
    wX !== nothing && rmul!(M, Diagonal(one(T) ./ sqrt.(wX)))
    return opnorm(M)
end

# ---------------------------------------------------------------------------
# InexactMap: exact F plus a controlled additive perturbation of its value and
# Jacobian. A_k v = F'(x)v + E v ; A_k^* θ = F'(x)^*θ + E^ᵀθ ; b_k = F(x) + e_b.
# ---------------------------------------------------------------------------
mutable struct InexactMap{T,FM<:VectorMap{T},MA<:AbstractMatrix{T},VY<:AbstractVector{T}} <: VectorMap{T}
    F::FM         # exact map (never mutated)
    E::MA         # Jacobian perturbation (m×n), refreshed each outer iteration
    eb::VY        # value perturbation (m), refreshed each outer iteration
end
function InexactMap(F::VectorMap{T}, n::Integer) where {T}
    m = rangedim(F)
    InexactMap{T,typeof(F),Matrix{T},Vector{T}}(F, zeros(T, m, n), zeros(T, m))
end
rangedim(im::InexactMap) = rangedim(im.F)
value!(b, im::InexactMap, x) = (value!(b, im.F, x); b .+= im.eb; b)
function applyJacobian!(out, im::InexactMap, v, x)
    applyJacobian!(out, im.F, v, x)
    mul!(out, im.E, v, one(eltype(out)), one(eltype(out)))   # out += E v
    return out
end
function applyAdjointJacobian!(out, im::InexactMap, θ, x)
    applyAdjointJacobian!(out, im.F, θ, x)
    mul!(out, transpose(im.E), θ, one(eltype(out)), one(eltype(out)))   # out += Eᵀ θ
    return out
end
applyAdjointHessian!(out, im::InexactMap, θ, v, x) = applyAdjointHessian!(out, im.F, θ, v, x)

# Scratch-taking forms (hot dual path): forward the buffer to the wrapped exact F
# so an InexactMap(QuadraticMap, …) adjoint is allocation-free, then add the
# perturbation. _dual_d! passes ws.qadj; modelhessvec! passes the model scratch.
# (E is constant, so F'' is unperturbed — the Hessian adjoint just delegates.)
function applyAdjointJacobian!(out, im::InexactMap, θ, x, scratch)
    applyAdjointJacobian!(out, im.F, θ, x, scratch)
    mul!(out, transpose(im.E), θ, one(eltype(out)), one(eltype(out)))   # out += Eᵀ θ
    return out
end
applyAdjointHessian!(out, im::InexactMap, θ, v, x, scratch) =
    applyAdjointHessian!(out, im.F, θ, v, x, scratch)

# Forward the anchor hook and the representation trait to the wrapped exact map, so
# a matrix-free F keeps its prepare!/factorization and its reuse gating under the
# inexactness path (where psi.F is the InexactMap, not prob.F).
prepare!(im::InexactMap, x) = (prepare!(im.F, x); im)
operator_actions_expensive(::Type{<:InexactMap{T,FM}}) where {T,FM} =
    operator_actions_expensive(FM)

# ---------------------------------------------------------------------------
# Audit record for one outer iteration.
# ---------------------------------------------------------------------------
struct InexactRecord{T}
    iter::Int
    hk::T; delta::T      # certified (conservative) h_k and the radius
    hpost::T             # stationarity of the FINAL injected model (hk must dominate)
    tolg::T; eg::T       # gradient bound and realized ‖g_k-∇f‖_X
    tolb::T; eb::T       # value bound and realized ‖b_k-F‖_Y
    tolj::T; eA::T       # jacobian bound and realized ‖A_k-F'‖_op
    pred::T; ared::T; cred::T; credbound::T   # Condition 2
end

# ---------------------------------------------------------------------------
# InexactOracle: controls + records the injected error.
# ---------------------------------------------------------------------------
mutable struct InexactOracle{T,SO<:SmoothObjective{T}}
    f::SO                # exact smooth objective (for the inexact gradient)
    kgrad::T; kval::T; kjac::T; kobj::T
    rho::T               # fraction of each tolerance actually injected, in (0,1)
    eta::T; zeta::T      # Condition 2 exponent params
    zeta_k::T; zfactor::T # forcing sequence ζ_k (multiplied by zfactor each outer iteration)
    # current-iteration scratch (filled during refresh / cred)
    tolg::T; tolb::T; tolj::T
    eg::T; ebn::T; eA::T; hpost::T
    records::Vector{InexactRecord{T}}
end
# `eta` must satisfy the paper's 0 < eta < min(eta1, 1-eta2); the default 1e-5 is
# below the default TROptions eta1 = 1e-4 (solve validates the actual pairing).
function InexactOracle(f::SmoothObjective{T};
                       kgrad=0.5, kval=0.5, kjac=0.5, kobj=1e2,
                       rho=0.5, eta=1e-5, zeta=1.5, zeta0=1.0, zfactor=0.5) where {T}
    zero(T) < rho <= one(T) || throw(ArgumentError("InexactOracle requires 0 < rho <= 1"))
    zeta > one(T) || throw(ArgumentError("InexactOracle requires zeta > 1"))
    zero(T) < zfactor < one(T) || throw(ArgumentError("InexactOracle requires 0 < zfactor < 1"))
    all(>(zero(T)), (kgrad, kval, kjac, kobj, eta, zeta0)) ||
        throw(ArgumentError("InexactOracle requires positive kgrad,kval,kjac,kobj,eta,zeta0"))
    z = zero(T)
    InexactOracle{T,typeof(f)}(f, T(kgrad), T(kval), T(kjac), T(kobj), T(rho),
                               T(eta), T(zeta), T(zeta0), T(zfactor),
                               z, z, z, z, z, z, z, InexactRecord{T}[])
end

# Deterministic unit-ish direction of length k (alternating, nonzero).
_dir(::Type{T}, k, shift) where {T} = T[((i + shift) % 3 == 0 ? -one(T) : one(T)) for i in 1:k]

# Set the Condition-1 tolerances from the current (h_k, Δ_k).
function set_tols!(o::InexactOracle{T}, hk, delta) where {T}
    o.tolg = o.kgrad * min(hk, delta)
    o.tolb = o.kval * min(hk, delta * delta)
    o.tolj = o.kjac * min(hk, delta)
    return o
end

# Inexact gradient g_k = ∇f(x) + e_g with ‖e_g‖_X = rho·tolg (recorded as `eg`).
function inject_gradient!(gcoord, o::InexactOracle{T}, X::HilbertSpace, x) where {T}
    gradient!(gcoord, o.f, x)
    d = _dir(T, length(x), 1)               # direction (coordinate)
    dp = similar(x); riesz!(dp, X, d)        # primal rep for the X-norm of the functional d
    nd = sqrt(max(inner(X, dp, dp), eps(T))) # ‖d‖_{X*} = ‖riesz(d)‖_X
    scale = o.rho * o.tolg / nd
    @. gcoord += scale * d
    o.eg = o.rho * o.tolg                     # realized ‖e_g‖_X (by construction)
    return gcoord
end

# Inexact linearization: set im.E (‖E‖_op = rho·tolj) and im.eb (‖e_b‖_Y = rho·tolb).
function inject_linearization!(im::InexactMap{T}, o::InexactOracle{T},
                               X::HilbertSpace, Y::HilbertSpace, x) where {T}
    m, n = size(im.E)
    # value perturbation e_b
    db = _dir(T, m, 0)
    nb = spnorm(Y, db)
    @. im.eb = (o.rho * o.tolb / nb) * db
    o.ebn = spnorm(Y, im.eb)                  # realized ‖b_k - F‖_Y
    # Jacobian perturbation E = c·(u vᵀ), rank-1 with controlled operator norm
    u = _dir(T, m, 0); v = _dir(T, n, 2)
    rv = similar(v); riesz!(rv, X, v)
    opn1 = spnorm(Y, u) * sqrt(max(inner(X, rv, rv), eps(T)))   # ‖u vᵀ‖_{L(X,Y)}
    c = o.rho * o.tolj / max(opn1, eps(T))
    mul!(im.E, reshape(c .* u, m, 1), reshape(v, 1, n))         # E = c u vᵀ
    o.eA = opnorm_metric(im.E, X, Y)          # realized ‖A_k - F'‖_op (computed, not assumed)
    return im
end

# Condition-2 computed reduction: cred = ared + e_c with |e_c| <= rho·credbound,
# credbound = kobj·[η·min(pred, ζ_k)]^ζ. Returns (cred, credbound).
function computed_reduction(o::InexactOracle{T}, ared, pred, iter) where {T}
    arg = o.eta * min(max(pred, zero(T)), o.zeta_k)
    credbound = o.kobj * arg^o.zeta
    sgn = isodd(iter) ? one(T) : -one(T)      # deterministic sign
    cred = ared + sgn * o.rho * credbound
    return cred, credbound
end

# Push the assembled audit record for this iteration.
function push_record!(o::InexactOracle{T}, iter, hk, delta, pred, ared, cred, credbound) where {T}
    push!(o.records, InexactRecord{T}(iter, hk, delta, o.hpost, o.tolg, o.eg, o.tolb, o.ebn,
                                      o.tolj, o.eA, pred, ared, cred, credbound))
    return o
end
