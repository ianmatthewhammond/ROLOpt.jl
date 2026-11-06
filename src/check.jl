# Part of ROLOpt.jl (MIT; see LICENSE). Portions are ported from Trilinos ROL,
# Copyright (2014) Sandia Corporation, under its BSD-style license (see LICENSE-ROL).

# Finite-difference derivative checks (ports of ROL `Objective::checkGradient`,
# `checkHessVec`, `checkHessSym`, `Constraint::checkApplyJacobian`,
# `checkAdjointConsistency`). Central differences are used so that for the
# quadratic test objectives the error is at roundoff level.
#
# Each `check_*` returns the minimum error over a range of step sizes, so tests
# can assert it is small.

# Template step sizes; each check_* converts them to the iterate's element type
# (`eltype(x)`) so the finite differences run in the working precision rather than
# silently promoting through Float64 (matters under BigFloat / Float32).
const _FD_STEPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

"""
    check_gradient(f, x, v) -> minerr

Compares the directional derivative `⟨∇f(x), v⟩` against a central finite
difference of `t -> f(x + t v)`.
"""
function check_gradient(f::SmoothObjective, x, v)
    g = gradient(f, x)
    exact = apply(g, v)
    Th = eltype(x)
    xp = similar(x)
    minerr = Th(Inf)
    for h0 in _FD_STEPS
        h = Th(h0)
        @. xp = x + h * v
        fp = value(f, xp)
        @. xp = x - h * v
        fm = value(f, xp)
        fd = (fp - fm) / (2h)
        minerr = min(minerr, abs(fd - exact))
    end
    return minerr
end

"""
    check_hessvec(f, x, v) -> minerr

Compares `∇²f(x) v` against a central finite difference of `∇f` along `v`.
"""
function check_hessvec(f::SmoothObjective, x, v)
    hv = hessVec(f, v, x)
    Th = eltype(x)
    gp = similar(x)
    gm = similar(x)
    xp = similar(x)
    fd = similar(x)
    minerr = Th(Inf)
    for h0 in _FD_STEPS
        h = Th(h0)
        @. xp = x + h * v
        gradient!(gp, f, xp)
        @. xp = x - h * v
        gradient!(gm, f, xp)
        @. fd = (gp - gm) / (2h)
        @. fd -= hv
        minerr = min(minerr, norm(fd))
    end
    return minerr
end

"""
    check_jacobian(F, x, v) -> minerr

Compares `F'(x) v` against a central finite difference of `F` along `v`.
"""
function check_jacobian(F::VectorMap{T}, x, v) where {T}
    prepare!(F, x)                      # assemble/factorize the linearization at x (no-op for plain maps)
    Jv = mapvalue(F, x)
    applyJacobian!(Jv, F, v, x)
    Th = eltype(x)
    fp = Vector{T}(undef, rangedim(F))
    fm = similar(fp)
    xp = similar(x)
    minerr = Th(Inf)
    for h0 in _FD_STEPS
        h = Th(h0)
        @. xp = x + h * v
        value!(fp, F, xp)
        @. xp = x - h * v
        value!(fm, F, xp)
        @. fp = (fp - fm) / (2h) - Jv
        minerr = min(minerr, norm(fp))
    end
    return minerr
end

"""
    check_adjoint(F, x, v, theta) -> err

Verifies the adjoint (transpose) identity `⟨F'(x) v, theta⟩ = ⟨v, F'(x)* theta⟩`
(metric-free coordinate pairing).
"""
function check_adjoint(F::VectorMap{T}, x, v, theta) where {T}
    prepare!(F, x)                      # assemble/factorize the linearization at x (no-op for plain maps)
    Jv = Vector{T}(undef, rangedim(F))
    applyJacobian!(Jv, F, v, x)
    Jatheta = similar(x)
    applyAdjointJacobian!(Jatheta, F, theta, x)
    return abs(dot(Jv, theta) - dot(v, Jatheta))
end

"""
    check_hess_sym(f, x, u, v) -> err

Verifies Hessian symmetry `⟨∇²f(x) u, v⟩ = ⟨∇²f(x) v, u⟩`.
"""
function check_hess_sym(f::SmoothObjective, x, u, v)
    Hu = hessVec(f, u, x)
    Hv = hessVec(f, v, x)
    return abs(apply(Hu, v) - apply(Hv, u))
end
