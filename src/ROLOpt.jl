module ROLOpt

# Inexact proximal trust region for structured nonsmooth / risk-averse
# optimization (Kouri, arXiv 2604.07216v1):
#
#     min_x  f(x) + suppA(F(x)) + phi(x)
#
# Implements the full algorithm: Algorithm 1 (TR-Risk loop, Cauchy or SPG step),
# Algorithm 2 (dual-SPG prox of psi_k), the risk envelopes, diagonal weighted
# (mass-matrix / probability) spaces, inexact evaluations (Conditions 1 & 2), and
# the (theta_k, F'') Lagrangian curvature term. C++ blueprints from Trilinos ROL
# (TypeP) are cited in each source file.

using LinearAlgebra
using Printf

include("spaces.jl")
include("objectives.jl")
include("usermaps.jl")
include("riskenvelope.jl")
include("check.jl")
include("psimodel.jl")
include("model.jl")
include("subsolvers.jl")
include("inexact.jl")
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

# User-selected F representations (matrix-free / cached Jacobian)
export OperatorMap, CachedJacobianMap, prepare!, operator_actions_expensive, moving_sample

# Risk envelope
export RiskEnvelope, IntervalEnvelope, ConvexCombinationEnvelope, SimplexEnvelope
export suppA, project!, projectdual, bound, initdual!, warmdual!

# Derivative checks
export check_gradient, check_hessvec, check_jacobian, check_adjoint, check_hess_sym

# psi_k model + Algorithm 2
export PsiModel, reanchor!, DualSPGResult, DualSPGOptions

# Model + subsolvers
export QuadraticModel, setmodel!, modelhessvec!, freeze_curvature!, Counters, pgstep!, hstat!, dcauchy!, dspg!, SpgParams
export dncg!, dbls!, NcgParams

# Inexactness
export InexactOracle, InexactMap, InexactRecord, opnorm_metric

# Trust region
export RiskProblem, TROptions, TRResult, CauchyParams, objective_value!, solve

end # module
