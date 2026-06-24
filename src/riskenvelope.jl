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
    # Validation lives in the inner constructor so the auto-generated outer
    # constructor (which would otherwise shadow validation for n::Int) is suppressed.
    function IntervalEnvelope{T}(a, b, n) where {T}
        a <= b || throw(ArgumentError("IntervalEnvelope requires a <= b"))
        n >= 1 || throw(ArgumentError("IntervalEnvelope requires n >= 1"))
        new{T}(a, b, Int(n))
    end
end
IntervalEnvelope(a::T, b::T, n::Integer=1) where {T} = IntervalEnvelope{T}(a, b, n)

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

`suppA(y) = sup_{theta in A} (theta,y)_Y` is evaluated metric-generically by the
continuous-knapsack (water-filling) solution and equals `R(y)` exactly when
`Y = WeightedSpace(probs)`. `projA` projects onto `A` in the Y metric `diag(w)`
via the KKT shift `thetaᵢ = clamp(yᵢ + nu·pᵢ/wᵢ, l, u)` with `nu` the root of
`Σ pᵢ thetaᵢ = 1` (uniform shift when `w = probs`; probability-weighted when
`w = 1`).
"""
struct ConvexCombinationEnvelope{T,V<:AbstractVector{T}} <: RiskEnvelope{T}
    lambda::T
    p::T
    probs::V
    l::T
    u::T
end
function ConvexCombinationEnvelope(lambda::T, p::T, probs::AbstractVector{T}) where {T}
    zero(T) <= lambda <= one(T) ||
        throw(ArgumentError("ConvexCombinationEnvelope requires 0 <= lambda <= 1"))
    zero(T) <= p < one(T) ||
        throw(ArgumentError("ConvexCombinationEnvelope requires 0 <= p < 1"))
    all(>(zero(T)), probs) ||
        throw(ArgumentError("ConvexCombinationEnvelope requires strictly positive probabilities"))
    isapprox(sum(probs), one(T); atol=sqrt(eps(T))) ||
        throw(ArgumentError("ConvexCombinationEnvelope requires sum(probs) ≈ 1"))
    l = one(T) - lambda
    u = (one(T) - lambda) + lambda / (one(T) - p)
    ConvexCombinationEnvelope{T,typeof(probs)}(lambda, p, probs, l, u)
end

# sigma_A(y) = max{ Σ wᵢ thetaᵢ yᵢ : Σ pᵢ thetaᵢ = 1, l ≤ thetaᵢ ≤ u } where
# wᵢ are the Y-metric weights (1 for Euclidean). This LP over a box ∩ hyperplane
# is a continuous knapsack: start at thetaᵢ = l, then raise thetaᵢ toward u in
# DESCENDING order of value density dᵢ = wᵢyᵢ/pᵢ until the budget Σ pᵢ thetaᵢ = 1
# is met (always feasible since the box capacity u-l = lambda/(1-p) ≥ lambda = 1-l).
# For Y = WeightedSpace(probs) this equals (1-lambda)E[y] + lambda·AVaR_p[y].
function suppA(A::ConvexCombinationEnvelope{T}, Y::HilbertSpace, y) where {T}
    p = A.probs; l = A.l; u = A.u
    w = proxweights(Y)
    n = length(y)
    val = zero(T); budget = one(T) - l            # budget = 1 - Σ pᵢ l = 1 - l
    # NOTE (Phase 3): the water-filling needs an ordering, so this evaluation
    # allocates `dens` (n) + the `sortperm` permutation — a BOUNDED O(n) + O(n log n)
    # allocation per call. It runs in `value(psi,·)` on the Cauchy/SPG model-value
    # path, so CVaR model evaluations are NOT zero-allocation (unlike box envelopes).
    # The envelope is deliberately kept immutable/reentrant (shared across solves);
    # making this alloc-free would require threading caller scratch through the
    # RiskEnvelope `suppA` interface, deferred. The dual-SPG inner kernels remain
    # alloc-free because `_dual_d!` uses `applyJacobian!`/`proxX!`/`project!`, not
    # `suppA`. The @allocated guards therefore target box (Interval) envelopes.
    dens = Vector{T}(undef, n)
    @inbounds for i in 1:n
        wi = w === nothing ? one(T) : w[i]
        val += wi * l * y[i]
        dens[i] = (wi * y[i]) / p[i]
    end
    ord = sortperm(dens; rev=true)
    cap = u - l
    @inbounds for i in ord
        budget <= zero(T) && break
        wi = w === nothing ? one(T) : w[i]
        delta = min(cap, budget / p[i])
        val += wi * delta * y[i]
        budget -= p[i] * delta
    end
    return val
end

# g(nu) = Σ pᵢ clamp(yᵢ + nu·(pᵢ/wᵢ), l, u) - 1, the projection budget residual
# (continuous, nondecreasing in nu). Module-level + dispatched on the weights so
# project! stays a closure-free, allocation-free bisection (it runs in the dual-SPG
# inner loop). `w === nothing` is the Euclidean (probability-weighted shift) case.
@inline function _cc_budget(p, ::Nothing, y, l::T, u::T, nu::T) where {T}
    s = zero(T)
    @inbounds @simd for i in eachindex(y)
        s += p[i] * clamp(y[i] + nu * p[i], l, u)
    end
    return s - one(T)
end
@inline function _cc_budget(p, w::AbstractVector, y, l::T, u::T, nu::T) where {T}
    s = zero(T)
    @inbounds @simd for i in eachindex(y)
        s += p[i] * clamp(y[i] + nu * (p[i] / w[i]), l, u)
    end
    return s - one(T)
end

function project!(out, A::ConvexCombinationEnvelope{T}, Y::HilbertSpace, y) where {T}
    p = A.probs
    w = proxweights(Y)
    l = A.l; u = A.u
    # Bracket the root of g(nu) (continuous, nondecreasing, and onto a neighborhood
    # of 0 since l ≤ 1 ≤ u and Σ pᵢ = 1), then bisect.
    nlo = zero(T); nhi = zero(T)
    if _cc_budget(p, w, y, l, u, nlo) > zero(T)
        nhi = nlo; nlo = -one(T)
        while _cc_budget(p, w, y, l, u, nlo) > zero(T); nlo *= 2; end
    else
        nhi = one(T)
        while _cc_budget(p, w, y, l, u, nhi) < zero(T); nhi *= 2; end
    end
    for _ in 1:200
        nu = (nlo + nhi) / 2
        if _cc_budget(p, w, y, l, u, nu) > zero(T); nhi = nu; else; nlo = nu; end
        nhi - nlo <= eps(T) * max(one(T), abs(nhi)) && break
    end
    nu = (nlo + nhi) / 2
    if w === nothing
        @inbounds @simd for i in eachindex(y)
            out[i] = clamp(y[i] + nu * p[i], l, u)
        end
    else
        @inbounds @simd for i in eachindex(y)
            out[i] = clamp(y[i] + nu * (p[i] / w[i]), l, u)
        end
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

# ---------------------------------------------------------------------------
# Probability-simplex envelope A = { theta : Σ thetaᵢ = 1, thetaᵢ ≥ 0 }
# ---------------------------------------------------------------------------
"""
    SimplexEnvelope{T}(n)

`A = { θ ∈ ℝⁿ : Σᵢ θᵢ = 1, θᵢ ≥ 0 }`, the probability simplex. Its support
function is the **maximum**: in the Y metric `diag(w)` (w = 1 for Euclidean),

    suppA(y) = sup_{θ∈A} (θ, y)_Y = max_i (wᵢ · yᵢ).

**With `Y = EuclideanSpace` (wᵢ ≡ 1) this is exactly `maximum(y)`** — the risk
measure ROLOpt needs to minimize a true (sampled) max, e.g. `max_i σ_vm(q_i)`.
Note the weighting: under a *weighted* Y, `suppA` is `max_i(wᵢ yᵢ)`, NOT
`max_i yᵢ`; for a sampled hard-max use Euclidean Y unless a weighting is intended.

`projA` is the Y-metric projection onto the simplex via the KKT shift
`θᵢ = max(yᵢ + ν/wᵢ, 0)` with `ν` the root of the budget `Σ θᵢ = 1` (Euclidean ⇒
the classic Duchi et al. projection). `MA = sqrt(max_i wᵢ)` (the simplex's extreme
points are the unit vertices `e_j`, so the Y-farthest is `‖e_j‖_Y = sqrt(w_j)`;
Euclidean ⇒ 1). This is the limiting CVaR set (`Σ θ = 1`, no upper bound), which
`ConvexCombinationEnvelope` cannot represent (it forbids `p ≥ 1`).
"""
struct SimplexEnvelope{T} <: RiskEnvelope{T}
    n::Int
    function SimplexEnvelope{T}(n::Integer) where {T}
        n >= 1 || throw(ArgumentError("SimplexEnvelope requires n >= 1"))
        new{T}(Int(n))
    end
end

function suppA(A::SimplexEnvelope{T}, Y::HilbertSpace, y) where {T}
    # The Σθ=1 budget couples all n entries, so length MUST match A.n — a mismatch
    # (e.g. an M_active envelope against an M_max-padded dual) would silently make the
    # initial dual infeasible. Guard it (cheap O(1) vs the O(n) loop).
    length(y) == A.n || throw(DimensionMismatch(
        "SimplexEnvelope(n=$(A.n)): suppA got y of length $(length(y))"))
    w = proxweights(Y)
    m = T(-Inf)
    @inbounds for i in eachindex(y)
        v = _wi(w, i) * y[i]
        v > m && (m = v)
    end
    return m
end

# g(nu) = Σ max(yᵢ + nu/wᵢ, 0) - 1, the projection budget residual (continuous,
# nondecreasing in nu). Module-level + dispatched on the weights so project! stays a
# closure-free, allocation-free bisection (it runs in the dual-SPG inner loop).
@inline function _simplex_budget(::Nothing, y, nu::T) where {T}
    s = zero(T)
    @inbounds @simd for i in eachindex(y)
        s += max(y[i] + nu, zero(T))
    end
    return s - one(T)
end
@inline function _simplex_budget(w::AbstractVector, y, nu::T) where {T}
    s = zero(T)
    @inbounds @simd for i in eachindex(y)
        s += max(y[i] + nu / w[i], zero(T))
    end
    return s - one(T)
end

function project!(out, A::SimplexEnvelope{T}, Y::HilbertSpace, y) where {T}
    (length(y) == A.n && length(out) == A.n) || throw(DimensionMismatch(
        "SimplexEnvelope(n=$(A.n)): project! got y/out of length $(length(y))/$(length(out))"))
    w = proxweights(Y)
    # Bracket the root of g(nu) (continuous, nondecreasing; a root exists since the
    # simplex is nonempty), then bisect — mirrors ConvexCombinationEnvelope's project!.
    nlo = zero(T); nhi = zero(T)
    if _simplex_budget(w, y, nlo) > zero(T)
        nhi = nlo; nlo = -one(T)
        while _simplex_budget(w, y, nlo) > zero(T); nlo *= 2; end
    else
        nhi = one(T)
        while _simplex_budget(w, y, nhi) < zero(T); nhi *= 2; end
    end
    for _ in 1:200
        nu = (nlo + nhi) / 2
        if _simplex_budget(w, y, nu) > zero(T); nhi = nu; else; nlo = nu; end
        nhi - nlo <= eps(T) * max(one(T), abs(nhi)) && break
    end
    nu = (nlo + nhi) / 2
    if w === nothing
        @inbounds @simd for i in eachindex(y)
            out[i] = max(y[i] + nu, zero(T))
        end
    else
        @inbounds @simd for i in eachindex(y)
            out[i] = max(y[i] + nu / w[i], zero(T))
        end
    end
    return out
end

function bound(A::SimplexEnvelope{T}, Y::HilbertSpace) where {T}
    w = proxweights(Y)
    wmax = w === nothing ? one(T) : maximum(w)
    return sqrt(wmax)
end
# theta = 1/n is feasible: Σ (1/n) = 1 and 1/n ≥ 0.
function initdual!(theta, A::SimplexEnvelope)
    length(theta) == A.n || throw(DimensionMismatch(
        "SimplexEnvelope(n=$(A.n)): initdual! got theta of length $(length(theta))"))
    fill!(theta, inv(eltype(theta)(A.n)))
    return theta
end
