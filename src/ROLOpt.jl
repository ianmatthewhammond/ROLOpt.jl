module ROLOpt

# Inexact proximal trust region for structured nonsmooth / risk-averse
# optimization (Kouri, arXiv 2604.07216v1):
#
#     min_x  f(x) + suppA(F(x)) + phi(x)
#
# Phase 1: Cauchy-only TR-Risk, exact evaluations, Euclidean spaces. See
# docs / the plan for the phased roadmap. C++ blueprints from Trilinos ROL
# (TypeP) are cited in each source file.

using LinearAlgebra
using Printf

include("spaces.jl")
include("objectives.jl")
include("riskenvelope.jl")
include("check.jl")
include("psimodel.jl")
include("model.jl")
include("subsolvers.jl")
include("trustregion.jl")

# Spaces
export HilbertSpace, EuclideanSpace, WeightedSpace
export inner, riesz!, metric!, proxweights, spnorm, apply

# Objectives
export SmoothObjective, ZeroObjective, DiagonalQuadratic, QuadraticObjective, TikhonovObjective
export VectorMap, AffineMap, QuadraticMap, rangedim, mapvalue
export ProxObjective, ZeroProx, L1Objective, BoundConstraint, BoxL1Objective
export value, value!, gradient, gradient!, hessVec, hessVec!
export applyJacobian!, applyAdjointJacobian!, applyAdjointHessian!, prox, prox!, proxX!

# Risk envelope
export RiskEnvelope, IntervalEnvelope, ConvexCombinationEnvelope
export suppA, project!, projectdual, bound, initdual!

# Derivative checks
export check_gradient, check_hessvec, check_jacobian, check_adjoint, check_hess_sym

# psi_k model + Algorithm 2
export PsiModel, reanchor!, DualSPGResult, DualSPGOptions

# Model + subsolvers
export QuadraticModel, setmodel!, modelhessvec!, Counters, pgstep!, hstat!, dcauchy!

# Trust region
export RiskProblem, TROptions, TRResult, CauchyParams, objective_value!, solve

end # module
