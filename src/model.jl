# Smooth quadratic trust-region model q_k of f, anchored at x_k:
#
#   q_k(x) = ½ (B_k (x - x_k), x - x_k)_X + (g_k, x - x_k)_X
#
# with g_k the primal gradient and B_k the model Hessian.
#
# B_k = HessVec_f                              (default; `F = nothing`)
# B_k = HessVec_f + (theta_k, F''(x_k) ·)      (curvature; `F` a VectorMap)
#
# Either is a legitimate bounded model Hessian (the convergence theory permits any
# bounded B_k). With the curvature term, theta_k is a FROZEN dual taken from the
# once-per-iteration stationarity prox (psi.theta after hstat) and held constant
# for the whole outer iteration -- the model-freezing invariant: the many proxes
# during the subproblem solve mutate psi.theta, but the model's frozen theta_k
# (and hence B_k) does not change. With `F = nothing` the model references no dual
# at all and the F'' path (applyAdjointHessian!) is never invoked.

struct QuadraticModel{T,SO,SX,VX,FM,VY}
    f::SO            # smooth objective (provides HessVec)
    X::SX            # X Hilbert space (for the Riesz map)
    xk::VX           # anchor x_k
    gk::VX           # primal gradient g_k = riesz_X(grad f(x_k)) (possibly inexact)
    F::FM            # VectorMap for the F'' curvature term, or `nothing`
    theta::VY        # frozen dual theta_k (Y-length), or empty
    htmp::VX         # scratch for the F'' contribution (coord X), or empty
end

# Default: f-only model (no F'' term).
function QuadraticModel(f::SmoothObjective{T}, X::HilbertSpace{T}, n::Integer) where {T}
    e = T[]
    QuadraticModel{T,typeof(f),typeof(X),Vector{T},Nothing,Vector{T}}(
        f, X, zeros(T, n), zeros(T, n), nothing, e, e)
end

# Curvature model: B_k includes (theta_k, F''(x_k) ·); m = dim(Y).
function QuadraticModel(f::SmoothObjective{T}, F::VectorMap{T}, X::HilbertSpace{T},
                        n::Integer, m::Integer) where {T}
    QuadraticModel{T,typeof(f),typeof(X),Vector{T},typeof(F),Vector{T}}(
        f, X, zeros(T, n), zeros(T, n), F, zeros(T, m), zeros(T, n))
end

"""
    setmodel!(m, xk, gcoord)

Re-anchor the model at `xk` with coordinate gradient `gcoord` (mapped to the
primal `g_k` via the Riesz map). Called once per outer iteration; the model is
constant thereafter.
"""
function setmodel!(m::QuadraticModel, xk, gcoord)
    copyto!(m.xk, xk)
    riesz!(m.gk, m.X, gcoord)
    return m
end

"""
    freeze_curvature!(m, theta)

Freeze the dual `theta_k` used by the F'' curvature term (no-op for an f-only
model). Called once per outer iteration, after the stationarity prox.
"""
freeze_curvature!(m::QuadraticModel{T,SO,SX,VX,Nothing}, theta) where {T,SO,SX,VX} = m
function freeze_curvature!(m::QuadraticModel{T,SO,SX,VX,<:VectorMap}, theta) where {T,SO,SX,VX}
    copyto!(m.theta, theta)
    return m
end

"""
    modelhessvec!(out, m, v) -> out

`B_k v` as a coordinate (dual) vector.
"""
modelhessvec!(out, m::QuadraticModel{T,SO,SX,VX,Nothing}, v) where {T,SO,SX,VX} =
    hessVec!(out, m.f, v, m.xk)
function modelhessvec!(out, m::QuadraticModel{T,SO,SX,VX,<:VectorMap}, v) where {T,SO,SX,VX}
    hessVec!(out, m.f, v, m.xk)                          # ∇²f(x_k) v
    applyAdjointHessian!(m.htmp, m.F, m.theta, v, m.xk)  # (theta_k, F''(x_k) v)
    out .+= m.htmp
    return out
end
