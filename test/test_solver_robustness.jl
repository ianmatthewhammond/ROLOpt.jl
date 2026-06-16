# Phase 1: solver robustness — failed-prox propagation and t0 != 1.

@testset "solver robustness" begin
    n = 3
    f = TikhonovObjective(1.0)
    F = AffineMap(reshape([1.0, 1.0, 1.0], 1, n), [0.5])   # active risk
    phi = ZeroProx{Float64}()
    A = IntervalEnvelope(0.0, 1.0)
    prob = RiskProblem(f, F, phi, A)
    xstar = fill(-1 / 6, n)

    @testset "failed Algorithm 2 prox is not certified as converged" begin
        # Starve the dual SPG so every prox hits its iteration cap.
        bad = DualSPGOptions{Float64}(maxit=0)
        res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=50, dualspg=bad))
        @test !res.converged
        @test res.status == :prox_failure
        @test res.counters.nproxfail > 0
    end

    @testset "t0 != 1 still converges to the solution" begin
        for t0 in (0.5, 2.0, 5.0)
            res = solve(prob, zeros(n); opts=TROptions{Float64}(t0=t0, maxit=500))
            @test res.converged
            @test res.status == :converged
            @test norm(res.x - xstar) < 1e-6
        end
    end

    @testset "dual-SPG iterations for h_k are counted in nspg" begin
        res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=50))
        # With an active risk term, the stationarity proxes run >= 1 dual iter each,
        # so nspg must exceed the dcauchy-only contribution — i.e. be positive and
        # at least the number of prox calls is reflected.
        @test res.counters.nspg > 0
        @test res.counters.nprox > res.iters     # includes the per-iterate h_k proxes
    end
end
