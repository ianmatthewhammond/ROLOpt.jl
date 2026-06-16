# Smooth quadratic trust-region model q_k of f, anchored at x_k:
#
#   q_k(x) = ½ (B_k (x - x_k), x - x_k)_X + (g_k, x - x_k)_X
#
# with g_k the primal gradient and B_k the model Hessian.
#
# PHASE 1: B_k v = HessVec_f(v) ONLY -- no F'' contribution. This is a legitimate
# bounded model Hessian (the convergence theory permits any bounded B_k), and it
# keeps the model independent of any dual variable, so it is trivially frozen for
# the whole outer iteration (see the model-freezing invariant in the plan). The
# (theta_k, F''(x_k) v) curvature term is deferred to Phase 2; this struct does
# not reference F at all, so the F'' path (applyAdjointHessian!) is never invoked.

struct QuadraticModel{T,SO,SX,VX}
    f::SO            # smooth objective (provides HessVec)
    X::SX            # X Hilbert space (for the Riesz map)
    xk::VX           # anchor x_k
    gk::VX           # primal gradient g_k = riesz_X(grad f(x_k)) (possibly inexact)
end

function QuadraticModel(f::SmoothObjective{T}, X::HilbertSpace{T}, n::Integer) where {T}
    QuadraticModel{T,typeof(f),typeof(X),Vector{T}}(f, X, zeros(T, n), zeros(T, n))
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
    modelhessvec!(out, m, v) -> out

`B_k v` as a coordinate (dual) vector. Phase 1: Hessian of `f` only.
"""
modelhessvec!(out, m::QuadraticModel, v) = hessVec!(out, m.f, v, m.xk)
