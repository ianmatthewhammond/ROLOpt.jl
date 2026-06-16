# Phase 2b: inexactness (paper Conditions 1 & 2), wired and audited. We do not
# infer the contract from convergence; we assert the realized per-iteration errors
# fall within the bounds defined by the POST-MODEL h_k the iteration actually used
# (a postcondition injection makes this hold for ANY kappa, including the paper's
# kappa = 1, and for rejected iterations after the radius shrinks), that
# |ared - cred| stays within the Condition-2 bound, and that the inexact run still
# converges to the exact solution.

@testset "inexactness (Conditions 1 & 2)" begin
    rng = MersenneTwister(31)

    @testset "oracle injection hits target norms (weighted metrics)" begin
        n, m = 5, 3
        X = WeightedSpace(rand(rng, n) .+ 0.5); Y = WeightedSpace(rand(rng, m) .+ 0.5)
        f = TikhonovObjective(1.0)
        F = AffineMap(randn(rng, m, n), randn(rng, m))
        o = InexactOracle(f; kgrad=1.0, kval=1.0, kjac=1.0, rho=0.5)
        im = ROLOpt.InexactMap(F, n)
        x = randn(rng, n)
        ROLOpt.set_tols!(o, 0.3, 0.8)
        @test o.tolg ≈ min(0.3, 0.8) && o.tolb ≈ min(0.3, 0.8^2) && o.tolj ≈ min(0.3, 0.8)
        g = zeros(n); ROLOpt.inject_gradient!(g, o, X, x)
        ed = g .- gradient(f, x); rp = similar(x); riesz!(rp, X, ed)
        @test sqrt(inner(X, rp, rp)) <= o.tolg + 1e-12            # ‖g_k-∇f‖_{X*} within bound
        ROLOpt.inject_linearization!(im, o, X, Y, x)
        @test opnorm_metric(im.E, X, Y) <= o.tolj + 1e-12         # ‖A_k-F'‖_op within bound
        bk = zeros(m); value!(bk, im, x); be = zeros(m); value!(be, F, x)
        @test spnorm(Y, bk .- be) <= o.tolb + 1e-12              # ‖b_k-F‖_Y within bound
    end

    @testset "contract holds every iteration + converges to exact (kappa = 1)" begin
        # kappa = 1 is the paper's value. The postcondition + reject refresh make
        # the contract hold for EVERY recorded iteration (accepted or rejected).
        cases = ((202, EuclideanSpace(Float64)),
                 (203, WeightedSpace(rand(MersenneTwister(210), 5) .+ 0.5)))
        for (seed, Xs) in cases
            srng = MersenneTwister(seed); n = 5
            f = TikhonovObjective(0.5)
            F = AffineMap(randn(srng, 1, n), [0.3])
            phi = L1Objective(fill(1e-2, n))
            A = IntervalEnvelope(0.0, 1.0)
            prob = RiskProblem(f, F, phi, A; X=Xs)
            kg = kv = kj = 1.0
            oracle = InexactOracle(f; kgrad=kg, kval=kv, kjac=kj, kobj=1e2, rho=0.5)
            res = solve(prob, zeros(n);
                        opts=TROptions{Float64}(maxit=2000, subsolver=:spg, delta0=10.0),
                        oracle=oracle)

            @test res.converged && res.hk <= 1e-8
            @test length(oracle.records) >= 1

            for r in oracle.records                       # EVERY iteration, incl. rejects
                @test r.eg <= kg * min(r.hk, r.delta)    + 1e-12    # ‖g_k-∇f‖_X
                @test r.eb <= kv * min(r.hk, r.delta^2)  + 1e-12    # ‖b_k-F‖_Y  (note Δ²)
                @test r.eA <= kj * min(r.hk, r.delta)    + 1e-12    # ‖A_k-F'‖_op
                @test abs(r.ared - r.cred) <= r.credbound + 1e-12   # Condition 2
                @test r.hk >= r.hpost - 1e-14            # certified metric dominates final model
            end
            @test any(r -> r.cred != r.ared, oracle.records)       # cred actually used
            @test oracle.records[end].credbound <= oracle.records[1].credbound + 1e-12

            re = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=2000, subsolver=:spg))
            @test norm(res.x - re.x) < 1e-6                        # exact minimizer
        end
    end

    @testset "rejected iterations refresh data for the new radius" begin
        # Force rejections with a tiny initial radius, then check Condition 1 holds
        # for every recorded (mostly rejected) iteration.
        srng = MersenneTwister(404); n = 4
        f = TikhonovObjective(0.5); F = AffineMap(randn(srng, 1, n), [0.4])
        prob = RiskProblem(f, F, L1Objective(fill(2e-2, n)), IntervalEnvelope(0.0, 1.0))
        oracle = InexactOracle(f; kgrad=1.0, kval=1.0, kjac=1.0, rho=0.5)
        res = solve(prob, zeros(n);
                    opts=TROptions{Float64}(maxit=2000, subsolver=:cauchy, delta0=1e-3, gamma3=2.0),
                    oracle=oracle)
        @test length(oracle.records) >= 5
        for r in oracle.records
            @test r.eg <= min(r.hk, r.delta)   + 1e-12
            @test r.eb <= min(r.hk, r.delta^2) + 1e-12
            @test r.eA <= min(r.hk, r.delta)   + 1e-12
        end
    end

    @testset "oracle constructor validation" begin
        f = TikhonovObjective(1.0)
        @test_throws ArgumentError InexactOracle(f; rho=0.0)
        @test_throws ArgumentError InexactOracle(f; rho=1.5)
        @test_throws ArgumentError InexactOracle(f; zeta=1.0)
        @test_throws ArgumentError InexactOracle(f; zfactor=1.0)
        @test_throws ArgumentError InexactOracle(f; zfactor=0.0)
        @test_throws ArgumentError InexactOracle(f; kgrad=-0.1)
        @test InexactOracle(f) isa InexactOracle
    end

    @testset "Condition 2 eta validation" begin
        n = 3
        f = TikhonovObjective(1.0); F = AffineMap(randn(rng, 1, n), [0.0])
        prob = RiskProblem(f, F, ZeroProx{Float64}(), IntervalEnvelope(0.0, 1.0))
        # default eta1 = 1e-4, so eta must be < 1e-4; eta = 1e-2 must be rejected.
        bad = InexactOracle(f; eta=1e-2)
        @test_throws ArgumentError solve(prob, zeros(n);
            opts=TROptions{Float64}(delta0=10.0), oracle=bad)
        ok = InexactOracle(f; eta=1e-5)
        @test solve(prob, zeros(n); opts=TROptions{Float64}(delta0=10.0, maxit=50), oracle=ok) isa TRResult
    end

    @testset "stall guard returns :stalled" begin
        # Force every step to be rejected (acceptance threshold above any achievable
        # ratio): Δ then shrinks each iteration until it collapses below the iterate
        # precision, exercising the stall guard (rather than spinning to maxit).
        srng = MersenneTwister(7); n = 5
        f = TikhonovObjective(0.5); F = AffineMap(randn(srng, 1, n), [0.3])
        prob = RiskProblem(f, F, L1Objective(fill(1e-2, n)), IntervalEnvelope(0.0, 1.0))
        res = solve(prob, zeros(n);
                    opts=TROptions{Float64}(maxit=5000, subsolver=:spg, delta0=10.0, eta1=2.0))
        @test res.status == :stalled
        @test !res.converged
        @test res.iters < 5000        # stopped early, did not spin to maxit
    end

    @testset "exact path unchanged when oracle = nothing" begin
        n = 4
        f = TikhonovObjective(1.0); F = AffineMap(randn(rng, 1, n), [0.2])
        prob = RiskProblem(f, F, L1Objective(fill(5e-2, n)), IntervalEnvelope(0.0, 1.0))
        opts = TROptions{Float64}(maxit=500, subsolver=:spg)
        r1 = solve(prob, zeros(n); opts=opts)
        r2 = solve(prob, zeros(n); opts=opts, oracle=nothing)
        @test r1.x == r2.x && r1.J == r2.J && r1.iters == r2.iters
        @test r1.counters.nprox == r2.counters.nprox
    end

    @testset "inexact solve requires delta0 > 0" begin
        n = 3
        f = TikhonovObjective(1.0); F = AffineMap(randn(rng, 1, n), [0.0])
        prob = RiskProblem(f, F, ZeroProx{Float64}(), IntervalEnvelope(0.0, 1.0))
        @test_throws ArgumentError solve(prob, zeros(n);
            opts=TROptions{Float64}(delta0=-1.0), oracle=InexactOracle(f))
    end
end
