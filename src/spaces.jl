# Hilbert space abstraction.
#
# Vectors are plain Julia arrays. A `HilbertSpace` supplies the metric: an inner
# product `(a,b)_X = aᵀ M a` and a Riesz map `riesz(g) = M⁻¹ g` that turns a dual
# / coordinate vector (e.g. a partial-derivative gradient) into its primal
# representative. This mirrors ROL's `Vector::dot` / `Vector::dual`.
#
# The metric-free duality pairing `apply(g, v) = ⟨g, v⟩ = sum(g .* v)` pairs a
# dual vector `g` with a primal vector `v` and never touches `M`. A key identity
# used throughout (and in ROL): for a coordinate gradient `g`,
#   inner(X, v, riesz(X, g)) == apply(g, v) == ⟨g, v⟩,
# so once a gradient is mapped to primal via `riesz`, the X-inner-product
# reproduces the duality pairing.
#
# Phase 1 uses only the Euclidean space (M = I): `inner = dot`, `riesz = copy`.
# Callbacks are stored as type parameters so they inline (no `::Function` field,
# no boxing).

"""
    HilbertSpace{T,I,R}

Real Hilbert space over element type `T`. `innerfn(a,b)` computes the inner
product; `rieszfn!(out,g)` writes the Riesz representative of dual vector `g`
into `out` and returns it.
"""
struct HilbertSpace{T<:Real,I,R}
    innerfn::I
    rieszfn!::R
end

"""
    EuclideanSpace(T) -> HilbertSpace{T}

Standard space with `M = I`: `inner = dot`, `riesz = copyto!` (identity).
"""
function EuclideanSpace(::Type{T}) where {T<:Real}
    innerfn = (a, b) -> dot(a, b)
    rieszfn! = (out, g) -> copyto!(out, g)
    return HilbertSpace{T,typeof(innerfn),typeof(rieszfn!)}(innerfn, rieszfn!)
end

"""
    WeightedSpace(weights) -> HilbertSpace

Diagonal metric `M = diag(weights)`: `(a,b) = Σ wᵢ aᵢ bᵢ`, `riesz(g)ᵢ = gᵢ / wᵢ`.
Used in Phase 2 for probability-weighted / mass-matrix spaces.
"""
function WeightedSpace(weights::AbstractVector{T}) where {T<:Real}
    w = weights
    innerfn = (a, b) -> begin
        s = zero(T)
        @inbounds @simd for i in eachindex(a, b, w)
            s += w[i] * a[i] * b[i]
        end
        s
    end
    rieszfn! = (out, g) -> (out .= g ./ w; out)
    return HilbertSpace{T,typeof(innerfn),typeof(rieszfn!)}(innerfn, rieszfn!)
end

@inline inner(X::HilbertSpace, a, b) = X.innerfn(a, b)
@inline riesz!(out, X::HilbertSpace, g) = X.rieszfn!(out, g)
@inline spnorm(X::HilbertSpace, a) = sqrt(max(inner(X, a, a), zero(eltype(a))))

# Metric-free duality pairing ⟨g, v⟩ for dual g, primal v.
@inline apply(g, v) = dot(g, v)

elty(::HilbertSpace{T}) where {T} = T
