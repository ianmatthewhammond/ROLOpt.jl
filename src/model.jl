# Smooth quadratic trust-region model q_k of f, anchored at x_k:
#
#   q_k(x) = ½ (B_k (x - x_k), x - x_k)_X + (g_k, x - x_k)_X
#
# with g_k the primal gradient and B_k the model Hessian.
#
# B_k = HessVec_f                                (default; `F = nothing`)
# B_k = HessVec_f + (theta_k, F''(x_k) ·)_Y      (curvature; `F` a VectorMap)
#
# The curvature term is the x-Hessian of the Lagrangian (theta_k, F(x))_Y, taken in
# the Y inner product: B_k v gains (theta_k, F''(x_k) v)_Y. Because the coordinate
# `applyAdjointHessian!` contracts F''(x)v against a coordinate dual, the stored
# multiplier is the METRIC dual M_Y·theta_k (see freeze_curvature!), so for weighted
# Y the metric is not dropped (it reduces to theta_k when M_Y = I).
#
# Either is a legitimate bounded model Hessian (the convergence theory permits any
# bounded B_k). With the curvature term, theta_k is a FROZEN dual taken from the
# once-per-iteration stationarity prox (psi.theta after hstat) and held constant
# for the whole outer iteration -- the model-freezing invariant: the many proxes
# during the subproblem solve mutate psi.theta, but the model's frozen theta_k
# (and hence B_k) does not change. With `F = nothing` the model references no dual
# at all and the F'' path (applyAdjointHessian!) is never invoked.

struct QuadraticModel{T,SO,SX,VX,FM,SY,VY}
    f::SO            # smooth objective (provides HessVec)
    X::SX            # X Hilbert space (for the Riesz map)
    xk::VX           # anchor x_k
    gk::VX           # primal gradient g_k = riesz_X(grad f(x_k)) (possibly inexact)
    F::FM            # VectorMap for the F'' curvature term, or `nothing`
    Y::SY            # Y Hilbert space (for the metric on the frozen dual), or `nothing`
    theta::VY        # frozen METRIC dual M_Y·theta_k (Y-length, coord), or empty
    htmp::VX         # scratch for the F'' contribution (coord X), or empty
    htmp2::VX        # scratch for the F'' adjoint gemv (coord X), or empty
end

# Default: f-only model (no F'' term).
function QuadraticModel(f::SmoothObjective{T}, X::HilbertSpace{T}, n::Integer) where {T}
    e = T[]
    QuadraticModel{T,typeof(f),typeof(X),Vector{T},Nothing,Nothing,Vector{T}}(
        f, X, zeros(T, n), zeros(T, n), nothing, nothing, e, e, e)
end

# Curvature model: B_k includes (theta_k, F''(x_k) ·)_Y; m = dim(Y).
function QuadraticModel(f::SmoothObjective{T}, F::VectorMap{T}, X::HilbertSpace{T},
                        Y::HilbertSpace{T}, n::Integer, m::Integer) where {T}
    QuadraticModel{T,typeof(f),typeof(X),Vector{T},typeof(F),typeof(Y),Vector{T}}(
        f, X, zeros(T, n), zeros(T, n), F, Y, zeros(T, m), zeros(T, n), zeros(T, n))
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
model). Called once per outer iteration, after the stationarity prox. Stores the
METRIC dual `M_Y·theta_k`, since `(theta, F''v)_Y` contracts the coordinate
`applyAdjointHessian!` against `M_Y·theta` (matching the Jacobian-adjoint path).
"""
freeze_curvature!(m::QuadraticModel{T,SO,SX,VX,Nothing}, theta) where {T,SO,SX,VX} = m
function freeze_curvature!(m::QuadraticModel{T,SO,SX,VX,<:VectorMap}, theta) where {T,SO,SX,VX}
    metric!(m.theta, m.Y, theta)                        # m.theta = M_Y · theta_k
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
    applyAdjointHessian!(m.htmp, m.F, m.theta, v, m.xk, m.htmp2)  # (theta_k, F''(x_k) v)_Y  (m.theta = M_Y·theta_k)
    out .+= m.htmp
    return out
end
