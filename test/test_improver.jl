# Phase 2b: the SPG improver subsolver (:spg). It must converge where Cauchy-only
# stalls on the nonsmooth CVaR risk term, reach the same stationary point on easy
# problems, and not change the exact-model / LASSO results.

@testset "SPG improver subsolver" begin
    rng = MersenneTwister(21)

    @testset "CVaR: :spg converges where :cauchy stalls" begin
        # A CVaR instance (seed 5) on which Cauchy-only is slow. J is convex
        # (Tikhonov + monotone CVaR ∘ convex F), so h_k <= htol certifies the
        # global minimizer.
        srng = MersenneTwister(5); n = 3; N = 4
        probs = (p = rand(srng, N) .+ 0.2; p ./ sum(p))
        Yp = WeightedSpace(probs)
        Q = [let Mt = randn(srng, n, n); (Mt'Mt) ./ n .* 0.05 end for _ in 1:N]
        F = QuadraticMap(Q, randn(srng, N, n), randn(srng, N))
        f = TikhonovObjective(0.2)
        A = ConvexCombinationEnvelope(0.75, 0.9, probs)
        prob = RiskProblem(f, F, ZeroProx{Float64}(), A; Y=Yp)

        rs = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=200, subsolver=:spg))
        rc = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=30, subsolver=:cauchy))
        @test rs.converged && rs.iters < 100     # improver certifies the (convex) optimum
        @test !rc.converged                      # Cauchy-only is nowhere near in 30 iters
        @test rs.hk < rc.hk                       # ...and far less stationary
    end

    @testset "improver agrees with Cauchy on easy problems" begin
        # active-risk affine/quadratic problem: both should find the same minimizer
        n = 4
        f = TikhonovObjective(1.0)
        F = AffineMap(randn(rng, 1, n), [0.3])
        phi = L1Objective(fill(5e-2, n))
        A = IntervalEnvelope(0.0, 1.0)
        prob = RiskProblem(f, F, phi, A)
        rc = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=1000, subsolver=:cauchy))
        rs = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=1000, subsolver=:spg))
        @test rc.converged && rs.converged
        @test norm(rc.x - rs.x) < 1e-6
    end

    @testset "LASSO reduction unchanged under :spg" begin
        n = 5
        a = 5 .* rand(rng, n) .+ 1e-2; b = 20 .* rand(rng, n) .- 10
        w = rand(rng, n); y = 10 .* rand(rng, n) .- 5
        f = DiagonalQuadratic(a, b)
        F = AffineMap(zeros(1, n), [0.0])
        prob = RiskProblem(f, F, L1Objective(w, y), IntervalEnvelope(0.0, 0.0))
        res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=5000, subsolver=:spg))
        xstar = @. (clamp(a * y + b, -w, w) - b) / a
        @test res.converged
        @test norm(res.x - xstar) < 1e3 * sqrt(eps())
    end

    @testset "subsolver validation" begin
        @test_throws ArgumentError TROptions{Float64}(subsolver=:bogus)
    end
end
