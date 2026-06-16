# Risk envelope A ⊂ Y: nonempty, closed, convex, bounded. Provides, w.r.t. the Y
# inner product, the support function suppA(Y,y) = sup_{theta in A} (theta,y)_Y,
# the metric projection projA(Y,y), the bound MA = sup_{theta in A} ‖theta‖_Y,
# and a feasible starting dual.
#
# The Y space is passed in so the envelope ops are consistent with the (diagonal)
# Y metric: support functions pick up the Y weights, and projections use the Y
# metric. This is the coupling that makes the risk term mesh-/measure-consistent.

abstract type RiskEnvelope{T<:Real} end

suppA(A::RiskEnvelope, Y, y) = throw(MethodError(suppA, (A, Y, y)))
project!(out, A::RiskEnvelope, Y, y) = throw(MethodError(project!, (out, A, Y, y)))
bound(A::RiskEnvelope, Y) = throw(MethodError(bound, (A, Y)))
initdual!(theta, A::RiskEnvelope) = throw(MethodError(initdual!, (theta, A)))

projectdual(A::RiskEnvelope, Y, y) = project!(similar(y), A, Y, y)

# Per-component Y weight (1 if Euclidean).
@inline _wi(w::Nothing, i) = true                 # multiplicative identity
@inline _wi(w::AbstractVector, i) = @inbounds w[i]

# ---------------------------------------------------------------------------
# Box envelope A = [a,b]^n
# ---------------------------------------------------------------------------
"""
    IntervalEnvelope{T}(a, b, n)

`A = [a, b]ⁿ ⊂ ℝⁿ`. In the Y metric `diag(w)` (w = 1 for Euclidean):
`suppA(y) = Σᵢ wᵢ·max(a·yᵢ, b·yᵢ)`, `projA(y) = clamp(y, a, b)` (box projection
is metric-independent), `MA = max(|a|,|b|)·sqrt(Σ wᵢ)`. For the paper's
sparse-control example `n = 1`, `[a,b] = [0,1]`: `suppA(y) = max(0, y)`.
"""
struct IntervalEnvelope{T} <: RiskEnvelope{T}
    a::T
    b::T
    n::Int
end
IntervalEnvelope(a::T, b::T) where {T} = IntervalEnvelope{T}(a, b, 1)
IntervalEnvelope(a::T, b::T, n::Integer) where {T} = IntervalEnvelope{T}(a, b, Int(n))

function suppA(A::IntervalEnvelope{T}, Y::HilbertSpace, y) where {T}
    w = proxweights(Y)
    s = zero(T)
    @inbounds @simd for i in eachindex(y)
        s += _wi(w, i) * max(A.a * y[i], A.b * y[i])
    end
    return s
end
project!(out, A::IntervalEnvelope, Y::HilbertSpace, y) = (@. out = clamp(y, A.a, A.b); out)
function bound(A::IntervalEnvelope{T}, Y::HilbertSpace) where {T}
    w = proxweights(Y)
    sw = w === nothing ? T(A.n) : sum(w)
    return max(abs(A.a), abs(A.b)) * sqrt(sw)
end
initdual!(theta, A::IntervalEnvelope) = (fill!(theta, clamp(zero(eltype(theta)), A.a, A.b)); theta)

# ---------------------------------------------------------------------------
# Convex-combination envelope: R = (1-lambda) E + lambda AVaR_p  (E+CVaR)
# ---------------------------------------------------------------------------
"""
    ConvexCombinationEnvelope(lambda, p, probs)

Risk envelope of `R(y) = (1-lambda)·E[y] + lambda·AVaR_p[y]` over a discrete
measure with probabilities `probs` (`E[y] = Σ pᵢ yᵢ`), namely
`A = { theta : Σ pᵢ thetaᵢ = 1, l ≤ thetaᵢ ≤ u }` with `l = 1-lambda`,
`u = (1-lambda) + lambda/(1-p)`. The natural Y space is `WeightedSpace(probs)`.

`suppA(y) = R(y)` is evaluated via `AVaR_p(y) = min_t t + 1/(1-p) Σ pᵢ (yᵢ-t)₊`.
`projA` projects onto `A` in the Y metric `diag(w)` via the KKT shift
`thetaᵢ = clamp(yᵢ + nu·pᵢ/wᵢ, l, u)` with `nu` the root of `Σ pᵢ thetaᵢ = 1`
(uniform shift when `w = probs`; probability-weighted shift when `w = 1`).
"""
struct ConvexCombinationEnvelope{T,V<:AbstractVector{T}} <: RiskEnvelope{T}
    lambda::T
    p::T
    probs::V
    l::T
    u::T
end
function ConvexCombinationEnvelope(lambda::T, p::T, probs::AbstractVector{T}) where {T}
    l = one(T) - lambda
    u = (one(T) - lambda) + lambda / (one(T) - p)
    ConvexCombinationEnvelope{T,typeof(probs)}(lambda, p, probs, l, u)
end

# AVaR_p(y) = min_t [ t + 1/(1-p) Σ pᵢ (yᵢ - t)₊ ]. The piecewise-linear inner
# objective is minimized at a breakpoint t = yᵢ, so scan the (sorted) samples.
function _avar(A::ConvexCombinationEnvelope{T}, y) where {T}
    p = A.probs
    c = one(T) / (one(T) - A.p)
    best = T(Inf)
    @inbounds for k in eachindex(y)
        t = y[k]
        acc = zero(T)
        @simd for i in eachindex(y)
            d = y[i] - t
            acc += p[i] * (d > zero(T) ? d : zero(T))
        end
        val = t + c * acc
        best = val < best ? val : best
    end
    return best
end

function suppA(A::ConvexCombinationEnvelope{T}, Y::HilbertSpace, y) where {T}
    Ey = zero(T)
    @inbounds @simd for i in eachindex(y)
        Ey += A.probs[i] * y[i]
    end
    return (one(T) - A.lambda) * Ey + A.lambda * _avar(A, y)
end

function project!(out, A::ConvexCombinationEnvelope{T}, Y::HilbertSpace, y) where {T}
    p = A.probs
    w = proxweights(Y)
    l = A.l; u = A.u
    # g(nu) = Σ pᵢ clamp(yᵢ + nu pᵢ/wᵢ, l, u) - 1, monotone increasing in nu.
    @inline ratio(i) = w === nothing ? p[i] : p[i] / @inbounds(w[i])
    function gval(nu)
        s = zero(T)
        @inbounds @simd for i in eachindex(y)
            s += p[i] * clamp(y[i] + nu * ratio(i), l, u)
        end
        return s - one(T)
    end
    # Bracket the root, then bisect (g is continuous, nondecreasing, and onto a
    # neighborhood of 0 since l ≤ 1 ≤ u and Σ pᵢ = 1).
    nlo = zero(T); nhi = zero(T)
    glo = gval(nlo)
    if glo > zero(T)
        nhi = nlo; ghi = glo; nlo = -one(T)
        glo = gval(nlo)
        while glo > zero(T); nlo *= 2; glo = gval(nlo); end
    else
        nhi = one(T); ghi = gval(nhi)
        while ghi < zero(T); nhi *= 2; ghi = gval(nhi); end
    end
    for _ in 1:200
        nu = (nlo + nhi) / 2
        g = gval(nu)
        if g > zero(T); nhi = nu; else; nlo = nu; end
        nhi - nlo <= eps(T) * max(one(T), abs(nhi)) && break
    end
    nu = (nlo + nhi) / 2
    @inbounds @simd for i in eachindex(y)
        out[i] = clamp(y[i] + nu * ratio(i), l, u)
    end
    return out
end

function bound(A::ConvexCombinationEnvelope{T}, Y::HilbertSpace) where {T}
    w = proxweights(Y)
    sw = w === nothing ? T(length(A.probs)) : sum(w)
    return max(abs(A.l), abs(A.u)) * sqrt(sw)
end
# theta ≡ 1 is feasible: Σ pᵢ·1 = 1 and l ≤ 1 ≤ u.
initdual!(theta, A::ConvexCombinationEnvelope) = (fill!(theta, one(eltype(theta))); theta)
