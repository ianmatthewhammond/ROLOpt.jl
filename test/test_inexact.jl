# Phase 2b: inexactness (paper Conditions 1 & 2), wired and audited. We do not
# infer the contract from convergence; we assert the realized per-iteration errors
# fall within the bounds defined by the POST-MODEL h_k the iteration actually used,
# and that |ared - cred| stays within the Condition-2 bound — then check the inexact
# run still converges and lands on the exact solution.

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

    # κ < 1 is the sound regime: the h_k fixed-point map has Lipschitz constant ~κ,
    # so it is a contraction (and the metric noise stays below htol) only for κ < 1
    # (consistent with the paper's κ_vs > 0). We test there.
    @testset "contract holds per iteration + converges to the exact solution" begin
        for Xs in (EuclideanSpace(Float64), WeightedSpace(rand(rng, 5) .+ 0.5))
            n = 5
            f = TikhonovObjective(0.5)
            F = AffineMap(randn(rng, 1, n), [0.3])
            phi = L1Objective(fill(1e-2, n))
            A = IntervalEnvelope(0.0, 1.0)
            prob = RiskProblem(f, F, phi, A; X=Xs)
            kg = kv = kj = 0.1
            htol = 1e-7                                   # above the κ=0.1 metric noise floor
            oracle = InexactOracle(f; kgrad=kg, kval=kv, kjac=kj, kobj=1e2, rho=0.5)
            res = solve(prob, zeros(n);
                        opts=TROptions{Float64}(maxit=500, subsolver=:spg, delta0=10.0, htol=htol),
                        oracle=oracle)

            @test res.converged                          # inexact run still reaches tolerance
            @test res.hk <= htol
            @test length(oracle.records) >= 1

            # Every iteration's data is within its own injection tolerance (ρ<1).
            for r in oracle.records
                @test r.eg <= r.tolg + 1e-12
                @test r.eb <= r.tolb + 1e-12
                @test r.eA <= r.tolj + 1e-12
                @test abs(r.ared - r.cred) <= r.credbound + 1e-12   # Condition 2
            end
            # Condition 1 against the POST-MODEL h_k, where h_k is meaningfully above
            # the metric noise floor (the fixed point is converged there).
            meaningful = filter(r -> r.hk > 10 * htol, oracle.records)
            @test length(meaningful) >= 1
            for r in meaningful
                @test r.eg <= kg * min(r.hk, r.delta)    + 1e-12    # ‖g_k-∇f‖_X
                @test r.eb <= kv * min(r.hk, r.delta^2)  + 1e-12    # ‖b_k-F‖_Y  (note Δ²)
                @test r.eA <= kj * min(r.hk, r.delta)    + 1e-12    # ‖A_k-F'‖_op
            end
            # the decision actually used cred (not the exact ared) on >=1 iteration
            @test any(r -> r.cred != r.ared, oracle.records)
            # forcing sequence ζ_k -> 0 shrinks the late Condition-2 bound
            @test oracle.records[end].credbound <= oracle.records[1].credbound + 1e-12

            # lands on the exact-solve minimizer (same prob, oracle = nothing)
            re = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=500, subsolver=:spg))
            @test norm(res.x - re.x) < 1e-6
        end
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
