# Risk envelope A ⊂ Y: nonempty, closed, convex, bounded. Provides the support
# function suppA(y) = sup_{theta∈A} (theta,y)_Y, the metric projection projA(y), the
# bound MA = sup_{theta∈A} ‖theta‖_Y, and a feasible starting dual.
#
# These are exactly the operations the paper's risk term needs:
#   * suppA  -> the nonsmooth value σ_A(F(x)) and the model value σ_A(ℓ_k(x))
#   * projA  -> the projection in Algorithm 2's dual spectral projected gradient
#
# Support function and projection are tied to the Y inner product. Phase 1 uses
# the Euclidean Y space; the IntervalEnvelope below is written for that case
# (probability-weighted geometry arrives with ConvexCombinationEnvelope, Phase 2).

abstract type RiskEnvelope{T<:Real} end

suppA(A::RiskEnvelope, y) = throw(MethodError(suppA, (A, y)))
project!(out, A::RiskEnvelope, y) = throw(MethodError(project!, (out, A, y)))
bound(A::RiskEnvelope) = throw(MethodError(bound, (A,)))
initdual!(theta, A::RiskEnvelope) = throw(MethodError(initdual!, (theta, A)))

projectdual(A::RiskEnvelope, y) = project!(similar(y), A, y)

"""
    IntervalEnvelope{T}(a, b, n)

`A = [a, b]ⁿ ⊂ ℝⁿ` (a per-component box, Euclidean Y). Then
`suppA(y) = Σᵢ max(a·yᵢ, b·yᵢ)`, `projA(y) = clamp(y, a, b)`,
`MA = sqrt(n)·max(|a|, |b|)`. For the paper's sparse-control example, `n = 1`,
`[a,b] = [0,1]`, giving `suppA(y) = max(0, y)` and `projA = clamp(y, 0, 1)`.
"""
struct IntervalEnvelope{T} <: RiskEnvelope{T}
    a::T
    b::T
    n::Int
end
IntervalEnvelope(a::T, b::T) where {T} = IntervalEnvelope{T}(a, b, 1)
IntervalEnvelope(a::T, b::T, n::Integer) where {T} = IntervalEnvelope{T}(a, b, Int(n))

function suppA(A::IntervalEnvelope{T}, y) where {T}
    s = zero(T)
    @inbounds @simd for i in eachindex(y)
        s += max(A.a * y[i], A.b * y[i])
    end
    return s
end
project!(out, A::IntervalEnvelope, y) = (@. out = clamp(y, A.a, A.b); out)
bound(A::IntervalEnvelope{T}) where {T} = sqrt(T(A.n)) * max(abs(A.a), abs(A.b))
# Feasible starting dual: the component of A closest to 0 (a valid theta ∈ A).
initdual!(theta, A::IntervalEnvelope) = (fill!(theta, clamp(zero(eltype(theta)), A.a, A.b)); theta)
