# Phase 1: solver robustness — failed-prox propagation and t0 != 1.

@testset "solver robustness" begin
    n = 3
    f = TikhonovObjective(1.0)
    F = AffineMap(reshape([1.0, 1.0, 1.0], 1, n), [0.5])   # active risk
    phi = ZeroProx{Float64}()
    A = IntervalEnvelope(0.0, 1.0)
    prob = RiskProblem(f, F, phi, A)
    xstar = fill(-1 / 6, n)

    @testset "grossly failed prox (maxit=0) is rejected, not treated as progress" begin
        # Starve the dual SPG so every prox is grossly unreliable (pgnorm = Inf).
        bad = DualSPGOptions{Float64}(maxit=0)
        J0 = objective_value!(zeros(1), prob, zeros(n))
        res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=50, dualspg=bad))
        @test !res.converged
        @test res.status == :prox_failure
        @test res.counters.nproxfail > 0
        # Gross steps are rejected: no spurious objective improvement, x unmoved.
        @test res.J ≈ J0 atol=1e-10
        @test norm(res.x) < 1e-10
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

    @testset "auto initial radius (delta0<=0) does not false-stall when f=0" begin
        # Regression for the auto-radius bug: with f = 0 the smooth gradient is zero,
        # so the OLD auto radius (= spnorm(X, dg)) collapsed to 0 and tripped the
        # stall guard on iteration 1 — a false stall. The active risk term gives a
        # nonzero prox-gradient step, so the radius must be positive and the solve
        # must make progress (the fix uses max(spnorm dg, spnorm step)).
        f0 = ZeroObjective{Float64}()
        F0 = AffineMap(reshape([1.0, 1.0, 1.0], 1, n), [0.5])   # ell = sum(x) + 0.5
        prob0 = RiskProblem(f0, F0, ZeroProx{Float64}(), IntervalEnvelope(0.0, 1.0))
        x0 = zeros(n)
        J0 = objective_value!(zeros(1), prob0, x0)              # max(0, 0.5) = 0.5
        res = solve(prob0, x0; opts=TROptions{Float64}(delta0=0.0, maxit=200))
        @test res.status != :stalled                           # the bug returned :stalled at iter 0
        @test res.iters > 0
        @test res.J < J0                                        # objective decreased
        @test norm(res.x .- x0) > 0                             # iterate moved off x0
        # J = max(0, sum(x)+0.5) is minimized to 0 on {sum(x) <= -0.5}; simple enough
        # to require convergence.
        @test res.converged && res.status == :converged
        @test res.J < 1e-6
    end
end
