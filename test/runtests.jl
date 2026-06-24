using ROLOpt
using Test
using LinearAlgebra
using Random

@testset "ROLOpt.jl" begin
    include("test_check.jl")
    include("test_riskenvelope.jl")
    include("test_simplex_max.jl")
    include("test_psimodel_prox.jl")
    include("test_cauchy_decrease.jl")
    include("test_lasso_reduce.jl")
    include("test_risk_finitedim_p1.jl")
    include("test_solver_robustness.jl")
    include("test_weighted.jl")
    include("test_improver.jl")
    include("test_ncg.jl")
    include("test_inexact.jl")
    include("test_curvature.jl")
    include("test_precision.jl")
    include("test_perf.jl")
    include("test_matfree.jl")
end
