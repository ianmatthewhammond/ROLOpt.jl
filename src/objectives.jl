# Problem-data interfaces + concrete finite-dimensional implementations.
#
# Derivative conventions (match ROL):
#   * `gradient!`, `hessVec!`, `applyAdjointJacobian!`, `applyAdjointHessian!`
#     return COORDINATE (dual) vectors (partial derivatives). Map to primal with
#     a space's `riesz!` where needed.
#   * `value!`, `applyJacobian!` return primal `Y` vectors.
#   * `prox!`, `project!` return primal vectors.
# All compute ops have in-place `!` primaries; thin allocating wrappers below.

# ----------------------------------------------------------------------------
# Smooth objective f: X -> R   (ROL Objective: value/gradient/hessVec)
# ----------------------------------------------------------------------------
abstract type SmoothObjective{T<:Real} end

value(f::SmoothObjective, x) = throw(MethodError(value, (f, x)))
gradient!(g, f::SmoothObjective, x) = throw(MethodError(gradient!, (g, f, x)))
hessVec!(hv, f::SmoothObjective, v, x) = throw(MethodError(hessVec!, (hv, f, v, x)))

gradient(f::SmoothObjective, x) = gradient!(similar(x), f, x)
hessVec(f::SmoothObjective, v, x) = hessVec!(similar(x), f, v, x)

"""ZeroObjective: f ≡ 0."""
struct ZeroObjective{T} <: SmoothObjective{T} end
ZeroObjective() = ZeroObjective{Float64}()
value(::ZeroObjective{T}, x) where {T} = zero(T)
gradient!(g, ::ZeroObjective, x) = fill!(g, zero(eltype(g)))
hessVec!(hv, ::ZeroObjective, v, x) = fill!(hv, zero(eltype(hv)))

"""
    DiagonalQuadratic(a, b)

`f(x) = Σ ½ aᵢ xᵢ² + bᵢ xᵢ`. Separable; matches ROL `test_01`'s smooth term.
"""
struct DiagonalQuadratic{T,V<:AbstractVector{T}} <: SmoothObjective{T}
    a::V
    b::V
end
function value(f::DiagonalQuadratic{T}, x) where {T}
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s += (f.a[i] * x[i] * x[i]) / 2 + f.b[i] * x[i]
    end
    return s
end
gradient!(g, f::DiagonalQuadratic, x) = (@. g = f.a * x + f.b; g)
hessVec!(hv, f::DiagonalQuadratic, v, x) = (@. hv = f.a * v; hv)

"""
    QuadraticObjective(H, b)

`f(x) = ½ xᵀ H x + bᵀ x` with symmetric `H`.
"""
struct QuadraticObjective{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <: SmoothObjective{T}
    H::M
    b::V
end
function value(f::QuadraticObjective{T}, x) where {T}
    return (dot(x, f.H, x)) / 2 + dot(f.b, x)
end
gradient!(g, f::QuadraticObjective, x) = (mul!(g, f.H, x); g .+= f.b; g)
hessVec!(hv, f::QuadraticObjective, v, x) = (mul!(hv, f.H, v); hv)

"""
    TikhonovObjective(tau)

`f(x) = (τ/2) Σ xᵢ²` (Euclidean ridge). Strongly convex for τ > 0.
"""
struct TikhonovObjective{T} <: SmoothObjective{T}
    tau::T
end
value(f::TikhonovObjective{T}, x) where {T} = (f.tau / 2) * dot(x, x)
gradient!(g, f::TikhonovObjective, x) = (@. g = f.tau * x; g)
hessVec!(hv, f::TikhonovObjective, v, x) = (@. hv = f.tau * v; hv)

# ----------------------------------------------------------------------------
# Smooth vector map F: X -> Y   (ROL Constraint: value/applyJacobian/adjoints)
# ----------------------------------------------------------------------------
abstract type VectorMap{T<:Real} end

value!(b, F::VectorMap, x) = throw(MethodError(value!, (b, F, x)))
applyJacobian!(out, F::VectorMap, v, x) = throw(MethodError(applyJacobian!, (out, F, v, x)))
applyAdjointJacobian!(out, F::VectorMap, theta, x) = throw(MethodError(applyAdjointJacobian!, (out, F, theta, x)))
# (theta, F''(x) v) — Phase 2 curvature term; Phase 1 must never call it.
applyAdjointHessian!(out, F::VectorMap, theta, v, x) = throw(MethodError(applyAdjointHessian!, (out, F, theta, v, x)))

"""rangedim(F): dimension of the output space Y."""
rangedim(F::VectorMap) = throw(MethodError(rangedim, (F,)))

mapvalue(F::VectorMap{T}, x) where {T} = value!(Vector{T}(undef, rangedim(F)), F, x)

"""
    AffineMap(A, c)

`F(x) = A x + c`, so `F'(x) = A` (constant) and `F'' = 0`.
"""
struct AffineMap{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <: VectorMap{T}
    A::M
    c::V
end
rangedim(F::AffineMap) = length(F.c)
value!(b, F::AffineMap, x) = (mul!(b, F.A, x); b .+= F.c; b)
applyJacobian!(out, F::AffineMap, v, x) = (mul!(out, F.A, v); out)
applyAdjointJacobian!(out, F::AffineMap, theta, x) = (mul!(out, transpose(F.A), theta); out)
applyAdjointHessian!(out, F::AffineMap, theta, v, x) = fill!(out, zero(eltype(out)))

"""
    QuadraticMap(Q, a, c)

Per output `j`: `Fⱼ(x) = ½ xᵀ Qⱼ x + aⱼᵀ x + cⱼ`, where `Q` is a vector of
symmetric matrices, `a` is `m×n` (row `j` = `aⱼᵀ`), `c ∈ ℝᵐ`.
`F'(x)` has rows `(Qⱼ x + aⱼ)ᵀ`; `(theta, F'' v) = Σⱼ thetaⱼ Qⱼ v`.
"""
struct QuadraticMap{T,MQ<:AbstractMatrix{T},MA<:AbstractMatrix{T},V<:AbstractVector{T}} <: VectorMap{T}
    Q::Vector{MQ}
    a::MA
    c::V
end
rangedim(F::QuadraticMap) = length(F.c)
function value!(b, F::QuadraticMap{T}, x) where {T}
    @inbounds for j in eachindex(F.c)
        b[j] = dot(x, F.Q[j], x) / 2 + dot(view(F.a, j, :), x) + F.c[j]
    end
    return b
end
function applyJacobian!(out, F::QuadraticMap{T}, v, x) where {T}
    @inbounds for j in eachindex(F.c)
        # (Qⱼ x + aⱼ) ⋅ v
        s = dot(view(F.a, j, :), v)
        s += dot(v, F.Q[j], x)
        out[j] = s
    end
    return out
end
function applyAdjointJacobian!(out, F::QuadraticMap{T}, theta, x) where {T}
    fill!(out, zero(T))
    tmp = similar(out)
    @inbounds for j in eachindex(F.c)
        thetaj = theta[j]                      # hoist scalar (avoid broadcast getindex under @.)
        mul!(tmp, F.Q[j], x)          # Qⱼ x
        aj = view(F.a, j, :)
        @. out += thetaj * (tmp + aj)
    end
    return out
end
function applyAdjointHessian!(out, F::QuadraticMap{T}, theta, v, x) where {T}
    fill!(out, zero(T))
    tmp = similar(out)
    @inbounds for j in eachindex(F.c)
        thetaj = theta[j]
        mul!(tmp, F.Q[j], v)          # Qⱼ v
        @. out += thetaj * tmp
    end
    return out
end

# ----------------------------------------------------------------------------
# Proximable convex term phi: X -> (-inf, +inf]   (ROL ProxObjective)
# ----------------------------------------------------------------------------
abstract type ProxObjective{T<:Real} end

value(phi::ProxObjective, x) = throw(MethodError(value, (phi, x)))
prox!(out, phi::ProxObjective, z, t) = throw(MethodError(prox!, (out, phi, z, t)))
prox(phi::ProxObjective, z, t) = prox!(similar(z), phi, z, t)

# Metric-aware prox in the X inner product:
#   prox^X_{t phi}(z) = argmin_x ½ (x-z)ᵀ M_X (x-z) + t phi(x).
# For a diagonal metric M_X = diag(w) and separable phi this is the per-component
# Euclidean prox with step t/wᵢ (the box-projection part of phi is metric-
# independent). The concrete prox! kernels accept a vector `t`, so we pass t/w.
@inline proxX!(out, phi::ProxObjective, ::HilbertSpace{T,Nothing}, z, t, teff) where {T} =
    prox!(out, phi, z, t)
@inline function proxX!(out, phi::ProxObjective, X::HilbertSpace{T,<:AbstractVector}, z, t, teff) where {T}
    @. teff = t / X.weights
    return prox!(out, phi, z, teff)
end

@inline _soft(u, k) = u - clamp(u, -k, k)   # soft-threshold: sign(u)·max(|u|-k,0)

"""ZeroProx: phi ≡ 0, prox = identity."""
struct ZeroProx{T} <: ProxObjective{T} end
ZeroProx() = ZeroProx{Float64}()
value(::ZeroProx{T}, x) where {T} = zero(T)
prox!(out, ::ZeroProx, z, t) = copyto!(out, z)

"""
    L1Objective(weights[, shift])

`phi(x) = Σ wᵢ |xᵢ - shiftᵢ|`. Port of ROL `l1Objective`:
`prox(z)ᵢ = zᵢ - clamp(zᵢ - shiftᵢ, -t wᵢ, t wᵢ)`.
"""
struct L1Objective{T,V<:AbstractVector{T}} <: ProxObjective{T}
    w::V
    shift::V
end
L1Objective(w::AbstractVector{T}) where {T} = L1Objective{T,typeof(w)}(w, zero(w))
function value(phi::L1Objective{T}, x) where {T}
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s += phi.w[i] * abs(x[i] - phi.shift[i])
    end
    return s
end
function prox!(out, phi::L1Objective, z, t)
    @. out = z - clamp(z - phi.shift, -t * phi.w, t * phi.w)
    return out
end

"""
    BoundConstraint(lo, hi)

Indicator of the box `[lo, hi]`; `prox = clamp(z, lo, hi)` (projection).
Port of ROL `Bounds`.
"""
struct BoundConstraint{T,V<:AbstractVector{T}} <: ProxObjective{T}
    lo::V
    hi::V
end
function value(phi::BoundConstraint{T}, x) where {T}
    @inbounds for i in eachindex(x)
        (x[i] < phi.lo[i] || x[i] > phi.hi[i]) && return T(Inf)
    end
    return zero(T)
end
prox!(out, phi::BoundConstraint, z, t) = (@. out = clamp(z, phi.lo, phi.hi); out)

"""
    BoxL1Objective(weights, lo, hi[, shift])

`phi(x) = Σ wᵢ |xᵢ - shiftᵢ| + ι_[lo,hi](x)` (the paper's sparse-control phi).
Separable, so the prox is the proven closed form
`prox(z)ᵢ = clamp(zᵢ - clamp(zᵢ - shiftᵢ, -t wᵢ, t wᵢ), loᵢ, hiᵢ)`
(soft-threshold then clamp) — NOT a generic composition of proxes.
"""
struct BoxL1Objective{T,V<:AbstractVector{T}} <: ProxObjective{T}
    w::V
    lo::V
    hi::V
    shift::V
end
BoxL1Objective(w::V, lo::V, hi::V) where {T,V<:AbstractVector{T}} =
    BoxL1Objective{T,V}(w, lo, hi, zero(w))
function value(phi::BoxL1Objective{T}, x) where {T}
    s = zero(T)
    @inbounds for i in eachindex(x)
        (x[i] < phi.lo[i] || x[i] > phi.hi[i]) && return T(Inf)
        s += phi.w[i] * abs(x[i] - phi.shift[i])
    end
    return s
end
function prox!(out, phi::BoxL1Objective, z, t)
    @. out = clamp(phi.shift + _soft(z - phi.shift, t * phi.w), phi.lo, phi.hi)
    return out
end
