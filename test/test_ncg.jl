# Phase 3d: the NCG improver subsolver (:ncg), a port of ROL dncg + dbls. Like :spg
# it must converge where Cauchy-only stalls on the CVaR risk term, agree on easy
# problems, leave the LASSO / exact results unchanged, never increase the model past
# the Cauchy point (so it stays in the trust region), and propagate gross prox
# failure rather than treating it as progress.

@testset "NCG improver subsolver (:ncg)" begin
    rng = MersenneTwister(41)

    @testset "CVaR: :ncg converges where :cauchy stalls" begin
        srng = MersenneTwister(5); n = 3; N = 4
        probs = (p = rand(srng, N) .+ 0.2; p ./ sum(p))
        Yp = WeightedSpace(probs)
        Q = [let Mt = randn(srng, n, n); (Mt'Mt) ./ n .* 0.05 end for _ in 1:N]
        F = QuadraticMap(Q, randn(srng, N, n), randn(srng, N))
        A = ConvexCombinationEnvelope(0.75, 0.9, probs)
        prob = RiskProblem(TikhonovObjective(0.2), F, ZeroProx{Float64}(), A; Y=Yp)
        rn = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=200, subsolver=:ncg))
        rc = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=30, subsolver=:cauchy))
        @test rn.converged && rn.iters < 100     # NCG certifies the (convex) optimum
        @test !rc.converged                      # Cauchy-only is nowhere near in 30 iters
        @test rn.hk < rc.hk
    end

    @testset ":ncg agrees with :cauchy on easy problems" begin
        n = 4
        F = AffineMap(randn(rng, 1, n), [0.3])
        prob = RiskProblem(TikhonovObjective(1.0), F, L1Objective(fill(5e-2, n)),
                           IntervalEnvelope(0.0, 1.0))
        rc = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=1000, subsolver=:cauchy))
        rn = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=1000, subsolver=:ncg))
        @test rc.converged && rn.converged
        @test norm(rc.x - rn.x) < 1e-6
    end

    @testset "LASSO reduction unchanged under :ncg" begin
        n = 5
        a = 5 .* rand(rng, n) .+ 1e-2; b = 20 .* rand(rng, n) .- 10
        w = rand(rng, n); y = 10 .* rand(rng, n) .- 5
        prob = RiskProblem(DiagonalQuadratic(a, b), AffineMap(zeros(1, n), [0.0]),
                           L1Objective(w, y), IntervalEnvelope(0.0, 0.0))
        res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=5000, subsolver=:ncg))
        xstar = @. (clamp(a * y + b, -w, w) - b) / a
        @test res.converged
        @test norm(res.x - xstar) < 1e3 * sqrt(eps())
    end

    @testset "subsolver / ncgType validation" begin
        @test TROptions{Float64}(subsolver=:ncg).subsolver === :ncg
        @test_throws ArgumentError TROptions{Float64}(subsolver=:bogus)
        @test_throws ArgumentError NcgParams{Float64}(ncgType=7)
        @test_throws ArgumentError NcgParams{Float64}(ncgType=-1)
        @test NcgParams{Float64}(ncgType=0).ncgType == 0
    end

    @testset "dncg! invariants: no worse than Cauchy, stays in TR" begin
        cp = CauchyParams{Float64}(1e-2, 0.1, 10.0, 10, 10, 1e-8)
        np = NcgParams{Float64}()
        for trial in 1:25
            n = rand(rng, 3:6)
            X = EuclideanSpace(Float64); Y = EuclideanSpace(Float64)
            f = DiagonalQuadratic(rand(rng, n) .+ 0.3, randn(rng, n))
            F = AffineMap(randn(rng, 1, n), randn(rng, 1))
            phi = trial % 2 == 0 ? L1Objective(rand(rng, n)) : ZeroProx{Float64}()
            psi = PsiModel(phi, IntervalEnvelope(0.0, 1.0), F, X, Y, n)
            x = randn(rng, n); reanchor!(psi, x)
            gc = gradient(f, x); dg = similar(x); riesz!(dg, X, gc)
            model = QuadraticModel(f, X, n); setmodel!(model, x, gc)
            sval = value(f, x); nval = value(psi, x)
            del = exp(randn(rng)); cnt = Counters()
            s = zeros(n); px = zeros(n); dwa = zeros(n); dwa1 = zeros(n)
            _, svalc, nvalc, _, _ = dcauchy!(s, model, psi, X, x, dg, del, 1.0,
                                             sval, nval, px, dwa, dwa1, cnt, cp)
            mC = svalc + nvalc
            y = x .+ s
            p1 = zeros(n); p2 = zeros(n); p3 = zeros(n); p4 = zeros(n); p5 = zeros(n)
            gmc = zeros(n); gmp = zeros(n)
            svaln, nvaln, _ = dncg!(y, model, psi, X, x, gc, del, svalc, nvalc,
                                    s, p1, p2, p3, p4, p5, dwa, gmc, gmp, cnt, np)
            @test svaln + nvaln <= mC + 1e-9                  # never increases the model
            @test spnorm(X, y .- x) <= del * (1 + 1e-8)       # stays inside the trust region
        end
    end

    @testset "1-D smooth: outward NCG direction at the TR boundary (dbls! tmax=0)" begin
        # f = ½x² - 10x (min at x=10), risk inert (A={0}, F=0). A small initial radius
        # puts the Cauchy point on the TR boundary with the NCG direction pointing
        # outward => alphaMax = 0 => dbls! receives tmax = 0. Must not NaN, and must
        # converge like :cauchy/:spg (regression for the 0/0 in the quadratic bound).
        prob = RiskProblem(DiagonalQuadratic([1.0], [-10.0]), AffineMap(zeros(1, 1), [0.0]),
                           ZeroProx{Float64}(), IntervalEnvelope(0.0, 0.0))
        rc = solve(prob, [0.0]; opts=TROptions{Float64}(delta0=1.0, maxit=50, subsolver=:cauchy))
        rn = solve(prob, [0.0]; opts=TROptions{Float64}(delta0=1.0, maxit=50, subsolver=:ncg))
        @test rn.converged && rn.status == :converged
        @test all(isfinite, rn.x) && isfinite(rn.J)
        @test isapprox(rn.x[1], 10.0; atol=1e-6)
        @test isapprox(rn.J, rc.J; atol=1e-8)
        # dbls! with an empty interval returns no step, no NaN.
        @test dbls!([1.0], [1.0], 0.5, 0.0, 1.0, -1.0, ZeroProx{Float64}(), [0.0], 0.0, Counters()) == (0.0, 0.0)
    end

    @testset ":ncg gross prox failure is rejected, not progress" begin
        n = 3
        F = AffineMap(reshape([1.0, 1.0, 1.0], 1, n), [0.5])   # active risk
        prob = RiskProblem(TikhonovObjective(1.0), F, ZeroProx{Float64}(), IntervalEnvelope(0.0, 1.0))
        bad = DualSPGOptions{Float64}(maxit=0)                 # every prox grossly unreliable
        J0 = objective_value!(zeros(1), prob, zeros(n))
        res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=50, subsolver=:ncg, dualspg=bad))
        @test !res.converged
        @test res.status == :prox_failure
        @test res.counters.nproxfail > 0
        @test res.J ≈ J0 atol=1e-10
        @test norm(res.x) < 1e-10
    end
end
