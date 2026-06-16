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

    @testset "dspg! local invariants: no worse than Cauchy, stays in TR" begin
        # Directly drive dcauchy! then dspg! and check the two claimed invariants
        # across random instances, envelopes, and trust-region radii.
        cp = CauchyParams{Float64}(1e-2, 0.1, 10.0, 10, 10, 1e-8)
        sp = SpgParams{Float64}()
        for trial in 1:25
            n = rand(rng, 3:6)
            X = EuclideanSpace(Float64); Y = EuclideanSpace(Float64)
            f = DiagonalQuadratic(rand(rng, n) .+ 0.3, randn(rng, n))
            F = AffineMap(randn(rng, 1, n), randn(rng, 1))
            phi = trial % 2 == 0 ? L1Objective(rand(rng, n)) : ZeroProx{Float64}()
            A = IntervalEnvelope(0.0, 1.0)
            psi = PsiModel(phi, A, F, X, Y, n)
            x = randn(rng, n); reanchor!(psi, x)
            gc = gradient(f, x); dg = similar(x); riesz!(dg, X, gc)
            model = QuadraticModel(f, X, n); setmodel!(model, x, gc)
            sval = value(f, x); nval = value(psi, x)
            del = exp(randn(rng))                      # radius in ~[0.1, 10]
            cnt = Counters()

            s = zeros(n); px = zeros(n); dwa = zeros(n); dwa1 = zeros(n)
            _, svalc, nvalc, _, _ = dcauchy!(s, model, psi, X, x, dg, del, 1.0,
                                             sval, nval, px, dwa, dwa1, cnt, cp)
            mC = svalc + nvalc                          # model value at the Cauchy point
            y = x .+ s
            gmodc = zeros(n); gmodp = zeros(n); st = zeros(n); pgi = zeros(n)
            svals, nvals, _ = dspg!(y, model, psi, X, x, gc, del,
                                    svalc, nvalc, gmodc, gmodp, st, pgi, dwa, px, cnt, sp)
            mS = svals + nvals                          # model value after the improver

            @test mS <= mC + 1e-10                      # never increases the model
            @test spnorm(X, y .- x) <= del * (1 + 1e-8) # stays inside the trust region
        end
    end
end
