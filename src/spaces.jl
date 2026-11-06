# Part of ROLOpt.jl (MIT; see LICENSE). Portions are ported from Trilinos ROL,
# Copyright (c) 2014 NTESS and the ROL contributors, BSD-3-Clause (see LICENSE-ROL).

# Hilbert space abstraction (diagonal metrics).
#
# Vectors are plain Julia arrays. A `HilbertSpace` supplies the metric: an inner
# product `(a,b)_X = aᵀ M a`, a Riesz map `riesz(g) = M⁻¹ g` (dual/coordinate ->
# primal), and the forward metric apply `metric(v) = M v` (primal -> dual).
# This mirrors ROL's `Vector::dot` / `Vector::dual`.
#
# SCOPE: the metric `M` is DIAGONAL (`M = I` for Euclidean, or `M = diag(w)`).
# A diagonal metric is what makes the separable prox / box projection closed-form,
# and it is exactly what the paper uses for mesh independence: element volumes for
# the piecewise-constant control space `X`, and probabilities for `Y = L²(Ω,P)`.
# (A consistent P1 mass matrix is non-diagonal and would make even a separable phi
# prox non-separable; practitioners use the lumped/diagonal metric for this reason.)
#
# Identities (used throughout, and in ROL):
#   inner(X, v, riesz(X, g)) == apply(g, v) == dot(g, v)        (Riesz <-> pairing)
#   (A v, theta)_Y == (v, A* theta)_X  with  A* theta = riesz_X(Aᵀ (M_Y theta))
# The metric-free duality pairing `apply(g, v) = dot(g, v)` never touches `M`.

"""
    HilbertSpace{T,W}

Real Hilbert space over `T` with a diagonal metric. `W = Nothing` is Euclidean
(`M = I`); `W <: AbstractVector{T}` is `M = diag(weights)`.
"""
struct HilbertSpace{T<:Real,W}
    weights::W
end

"""Euclidean space (`M = I`)."""
EuclideanSpace(::Type{T}) where {T<:Real} = HilbertSpace{T,Nothing}(nothing)

"""Diagonal-metric space `M = diag(weights)` (e.g. element volumes, probabilities)."""
function WeightedSpace(weights::AbstractVector{T}) where {T<:Real}
    all(>(zero(T)), weights) ||
        throw(ArgumentError("WeightedSpace requires strictly positive weights"))
    HilbertSpace{T,typeof(weights)}(weights)
end

# Inner product (a,b)_X = aᵀ M b
@inline inner(::HilbertSpace{T,Nothing}, a, b) where {T} = dot(a, b)
@inline function inner(X::HilbertSpace{T,<:AbstractVector}, a, b) where {T}
    w = X.weights
    s = zero(T)
    @inbounds @simd for i in eachindex(a, b, w)
        s += w[i] * a[i] * b[i]
    end
    return s
end

# Riesz map M⁻¹ g (dual/coordinate -> primal)
@inline riesz!(out, ::HilbertSpace{T,Nothing}, g) where {T} = copyto!(out, g)
@inline riesz!(out, X::HilbertSpace{T,<:AbstractVector}, g) where {T} = (out .= g ./ X.weights; out)

# Forward metric M v (primal -> dual/coordinate)
@inline metric!(out, ::HilbertSpace{T,Nothing}, v) where {T} = copyto!(out, v)
@inline metric!(out, X::HilbertSpace{T,<:AbstractVector}, v) where {T} = (out .= X.weights .* v; out)

# Diagonal of M for separable prox / projection scaling (`nothing` => all ones).
@inline proxweights(::HilbertSpace{T,Nothing}) where {T} = nothing
@inline proxweights(X::HilbertSpace{T,<:AbstractVector}) where {T} = X.weights

@inline spnorm(X::HilbertSpace, a) = sqrt(max(inner(X, a, a), zero(eltype(a))))

# Metric-free duality pairing ⟨g, v⟩ for dual g, primal v.
@inline apply(g, v) = dot(g, v)

elty(::HilbertSpace{T}) where {T} = T
