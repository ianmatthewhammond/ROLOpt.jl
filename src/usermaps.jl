# User-selected matrix-free / cached-Jacobian risk maps. The user CHOOSES the
# representation by the VectorMap type they pass; ROLOpt never infers whether to
# assemble A_k. Both types are plain VectorMaps — the solver and Algorithm 2 touch
# them only through the operator-action interface (value!, applyJacobian!,
# applyAdjointJacobian!, optional applyAdjointHessian!, rangedim) plus the prepare!
# anchor hook. See docs/design/matrix_free_F.md.
#
# CONVENTION (the #1 pitfall): applyJacobian! returns a PRIMAL Y vector (A v);
# applyAdjointJacobian! returns the COORDINATE X vector Aᵀθ — the BARE transpose,
# NO metric. The framework forms the true metric adjoint riesz_X(Aᵀ(M_Y θ)) itself.
# Verify a JVP/VJP pair with check_jacobian / check_adjoint before solving.

# ───────────────────────── OperatorMap: pure matrix-free ─────────────────────
# No stored A_k; every apply is a user callback (e.g. a PDE tangent/adjoint solve).
# Callbacks may return anything (mutating callbacks often return nothing), so each
# method forces the output-buffer return.
#
# Keyword callbacks (bang-free names; the functions you pass can be `!`-named):
#   value(b, x)        -> b = F(x)              (primal Y)
#   jvp(out, v, x)     -> out = F'(x) v          (primal Y)
#   vjp(out, th, x)    -> out = F'(x)ᵀ th        (coord X, bare transpose)
#   ahp(out, th, v, x) -> out = (th, F''(x) v)   (optional; coord X)
#   prepare(x)         -> (optional) factorize/cache the frozen linearization at x
# `moving_sample=true` marks an F whose output sample set re-indexes across anchors
# (e.g. a moving-cut sampled-stress map); then reanchor! resets the dual theta.
struct OperatorMap{T,FV,FJ,FA,FH,FP} <: VectorMap{T}
    m::Int
    fval::FV
    fjvp::FJ
    fvjp::FA
    fahp::FH      # Nothing if no curvature callback
    fprep::FP     # Nothing if no prepare callback
    moving_sample::Bool
end
# Keyword names are bang-free (`vjp!=f` would lex as `vjp != f`); the callbacks
# themselves can still be `!`-named functions.
function OperatorMap{T}(; m::Integer, value, jvp, vjp, ahp=nothing, prepare=nothing,
                        moving_sample::Bool=false) where {T}
    OperatorMap{T,typeof(value),typeof(jvp),typeof(vjp),typeof(ahp),typeof(prepare)}(
        Int(m), value, jvp, vjp, ahp, prepare, moving_sample)
end

rangedim(F::OperatorMap) = F.m
moving_sample(F::OperatorMap) = F.moving_sample
value!(b, F::OperatorMap, x)                = (F.fval(b, x); b)
applyJacobian!(out, F::OperatorMap, v, x)   = (F.fjvp(out, v, x); out)
applyAdjointJacobian!(out, F::OperatorMap, theta, x) = (F.fvjp(out, theta, x); out)
function applyAdjointHessian!(out, F::OperatorMap, theta, v, x)
    F.fahp === nothing && throw(ArgumentError(
        "OperatorMap has no ahp!; set use_curvature=false or supply ahp!"))
    F.fahp(out, theta, v, x); out
end
prepare!(F::OperatorMap, x) = (F.fprep === nothing || F.fprep(x); F)

# Matrix-free: every apply is a solve, so the Phase-2 line-search reuse pays off.
operator_actions_expensive(::Type{<:OperatorMap}) = true

# ────────────── CachedJacobianMap: assemble A_k = F'(x_k) once per anchor ─────
# Nonlinear F whose Jacobian is storable; the user supplies assemble!(Ak, x) that
# builds A_k = F'(x). `prepare!` runs it once per anchor; applies are cached gemvs.
# Distinct from AffineMap (whose A is fixed): here A_k is the local linearization,
# rebuilt whenever x_k moves. `value!` is the TRUE nonlinear F (used for b_k); the
# linearization cache (Ak) is owned by prepare! and must not be touched by value!.
mutable struct CachedJacobianMap{T,MA<:AbstractMatrix{T},FV,FAS,FH} <: VectorMap{T}
    m::Int
    Ak::MA
    fval::FV
    fassemble::FAS
    fahp::FH      # Nothing if no curvature callback
end
function CachedJacobianMap{T}(; m::Integer, Ak::AbstractMatrix{T},
                              value, assemble, ahp=nothing) where {T}
    size(Ak, 1) == Int(m) || throw(ArgumentError(
        "CachedJacobianMap: size(Ak,1)=$(size(Ak, 1)) must equal m=$(Int(m)) (= rangedim)"))
    CachedJacobianMap{T,typeof(Ak),typeof(value),typeof(assemble),typeof(ahp)}(
        Int(m), Ak, value, assemble, ahp)
end

rangedim(F::CachedJacobianMap) = F.m
# assemble! may mutate Ak in place (return nothing) or return a new matrix of the
# same shape; a wrong shape becomes a clear error here, not a later DimensionMismatch.
function prepare!(F::CachedJacobianMap, x)
    newAk = F.fassemble(F.Ak, x)
    if newAk !== nothing
        size(newAk) == size(F.Ak) || throw(ArgumentError(
            "CachedJacobianMap assemble! returned size $(size(newAk)), expected $(size(F.Ak))"))
        F.Ak = newAk
    end
    return F
end
value!(b, F::CachedJacobianMap, x)              = (F.fval(b, x); b)
applyJacobian!(out, F::CachedJacobianMap, v, x) = (mul!(out, F.Ak, v); out)
applyAdjointJacobian!(out, F::CachedJacobianMap, theta, x) = (mul!(out, transpose(F.Ak), theta); out)
function applyAdjointHessian!(out, F::CachedJacobianMap, theta, v, x)
    F.fahp === nothing && throw(ArgumentError(
        "CachedJacobianMap has no ahp!; set use_curvature=false or supply ahp!"))
    F.fahp(out, theta, v, x); out
end
# operator_actions_expensive inherits the VectorMap default (false): applies are gemvs.
