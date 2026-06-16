using ROLOpt
using Test
using LinearAlgebra
using Random

@testset "ROLOpt.jl (Phase 1: Cauchy-only TR-Risk)" begin
    include("test_check.jl")
    include("test_riskenvelope.jl")
    include("test_psimodel_prox.jl")
    include("test_cauchy_decrease.jl")
    include("test_lasso_reduce.jl")
    include("test_risk_finitedim_p1.jl")
    include("test_solver_robustness.jl")
    include("test_weighted.jl")
    include("test_improver.jl")
    include("test_inexact.jl")
end
