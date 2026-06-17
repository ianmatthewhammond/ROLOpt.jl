# Phase 3c: generic precision (T<:Real). The check_* finite differences now run in
# eltype(x) (no silent Float64 promotion), and the solver's T-parametrization is
# exercised under BigFloat (tight tol, beyond Float64) and Float32 (type-stable +
# loose-tol convergence with precision-scaled tolerances — its defaults like
# htol=1e-8 / dual tol=1e-10 sit below eps(Float32) ≈ 1.2e-7).

@testset "generic precision (BigFloat + Float32)" begin

    @testset "FD derivative checks run in the working precision" begin
        for (T, gtol) in ((BigFloat, BigFloat(1e-25)), (Float32, 1.0f-2))
            rng = MersenneTwister(2026)
            n = 4; m = 3
            f = QuadraticObjective((let M = randn(rng, n, n); T.(M'M) end) + I, T.(randn(rng, n)))
            x = T.(randn(rng, n)); v = T.(randn(rng, n)); u = T.(randn(rng, n))
            @test check_gradient(f, x, v) < gtol
            @test check_hessvec(f, x, v) < gtol
            @test check_hess_sym(f, x, u, v) < gtol
            Q = [(let M = randn(rng, n, n); T.(M'M) ./ n end) for _ in 1:m]
            F = QuadraticMap(Q, T.(randn(rng, m, n)), T.(randn(rng, m)))
            @test check_jacobian(F, x, v) < gtol
            @test check_adjoint(F, x, v, T.(randn(rng, m))) < gtol
        end
    end

    # min ½‖x‖² + max(0, sum(x)+0.5), solution x* = -1/6 · ones (sum(x*) = -0.5).
    risk1(::Type{T}, n) where {T} = RiskProblem(TikhonovObjective(one(T)),
        AffineMap(ones(T, 1, n), T[0.5]), ZeroProx{T}(), IntervalEnvelope(zero(T), one(T)))

    @testset "BigFloat solve converges beyond Float64 precision" begin
        T = BigFloat; n = 3
        opts = TROptions{T}(htol=T(1e-20), maxit=300,
                            dualspg=DualSPGOptions{T}(tol=T(1e-30), gmin=T(1e-40)))
        res = solve(risk1(T, n), zeros(T, n); opts=opts)
        @test res.converged && res.status == :converged
        @test res.hk <= T(1e-20)
        @test norm(res.x .- (-one(T) / 6)) < T(1e-18)
    end

    @testset "Float32 solve: type-stable + loose-tol convergence" begin
        T = Float32; n = 3
        opts = TROptions{T}(htol=1.0f-3, maxit=200, dualspg=DualSPGOptions{T}(tol=1.0f-5))
        res = @inferred solve(risk1(T, n), zeros(T, n); opts=opts)
        @test res isa TRResult{Float32,Vector{Float32}}
        @test res.converged && res.status == :converged
        @test norm(res.x .- (-one(T) / 6)) < 1.0f-2
    end
end
