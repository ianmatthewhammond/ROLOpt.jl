# Weighted (diagonal-metric) consistency — tests designed to FAIL if any op
# silently assumes the Euclidean metric. Uses nonuniform weights throughout.

# Independent weighted stationarity residual, built from scratch at x.
function indep_hstat_w(prob, x, t0)
    n = length(x)
    psi = PsiModel(prob.phi, prob.A, prob.F, prob.X, prob.Y, n)
    reanchor!(psi, x)
    g = gradient(prob.f, x); dg = similar(x); riesz!(dg, prob.X, g)
    z = x .- t0 .* dg
    p = similar(x); prox!(p, psi, z, t0)
    return spnorm(prob.X, p .- x) / t0
end

@testset "weighted-metric consistency" begin
    rng = MersenneTwister(11)
    n, m = 6, 4
    wX = rand(rng, n) .+ 0.4
    wY = rand(rng, m) .+ 0.4
    X = WeightedSpace(wX); Y = WeightedSpace(wY)

    @testset "Riesz / metric / pairing identities" begin
        for _ in 1:50
            v = randn(rng, n); g = randn(rng, n)
            dgv = similar(v); riesz!(dgv, X, g)
            @test inner(X, v, dgv) ≈ dot(g, v)           # inner(v, riesz g) = ⟨g,v⟩
            mv = similar(v); metric!(mv, X, v)
            rmv = similar(v); riesz!(rmv, X, mv)
            @test rmv ≈ v                                # riesz ∘ metric = id
            @test inner(X, v, v) ≈ sum(wX .* v .^ 2)
        end
    end

    @testset "weighted prox optimality (L1)" begin
        lam = rand(rng, n) .+ 0.05
        phi = L1Objective(lam)
        for _ in 1:30
            z = randn(rng, n); t = 0.2 + rand(rng)
            out = similar(z); teff = similar(z)
            proxX!(out, phi, X, z, t, teff)
            # closed form: weighted prox = soft-threshold with step t·λ/w
            cf = @. sign(z) * max(abs(z) - t * lam / wX, 0)
            @test out ≈ cf atol=1e-12
            # subgradient optimality: w(out-z)/t = -λ·sign(out) on the support
            for i in 1:n
                if abs(out[i]) > 1e-9
                    @test wX[i] * (out[i] - z[i]) / t ≈ -lam[i] * sign(out[i]) atol=1e-9
                end
            end
            # the metric matters: differs from the Euclidean prox when w ≠ 1
            euc = @. sign(z) * max(abs(z) - t * lam, 0)
            @test !isapprox(out, euc; atol=1e-3)
        end
    end

    @testset "metric adjoint consistency (A v, θ)_Y = (v, A* θ)_X" begin
        Amat = randn(rng, m, n); c = randn(rng, m)
        F = AffineMap(Amat, c)
        x = randn(rng, n)
        for _ in 1:50
            v = randn(rng, n); theta = randn(rng, m)
            Jv = similar(c); applyJacobian!(Jv, F, v, x)
            # A* θ = riesz_X(Aᵀ (M_Y θ))
            myt = similar(theta); metric!(myt, Y, theta)
            adj = similar(v); applyAdjointJacobian!(adj, F, myt, x)
            Astar = similar(v); riesz!(Astar, X, adj)
            @test inner(Y, Jv, theta) ≈ inner(X, v, Astar) atol=1e-10
        end
    end

    @testset "weighted-X problem converges (metric-natural f)" begin
        # f(x) = (tau/2)(x,x)_X  => coord grad = tau·wX·x, hessVec = tau·wX·v, so the
        # model Hessian B_k = M_X⁻¹H = tau·I is well-conditioned in the X metric
        # (mirrors the paper's (tau/2)‖x‖_X² and its mesh independence). The Cauchy
        # method then converges as fast under the weighted metric as the Euclidean.
        f = DiagonalQuadratic(0.5 .* wX, zeros(n))
        F = AffineMap(randn(rng, 1, n), [0.4])           # scalar Y, active risk
        phi = L1Objective(fill(1e-2, n))
        A = IntervalEnvelope(0.0, 1.0)
        prob = RiskProblem(f, F, phi, A; X=X)            # weighted X, Euclidean scalar Y
        res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=500))
        @test res.converged
        @test res.hk <= 1e-8
        @test indep_hstat_w(prob, res.x, 1.0) < 1e-7
    end

    @testset "CVaR on weighted Y: progress + h_k consistency" begin
        # Cauchy-only is slow on the nonsmooth CVaR risk term (the NCG/SPG improver
        # is the next Phase-2 piece), so we validate the metric-aware machinery via
        # (i) substantial progress and (ii) an independent weighted stationarity
        # residual that MATCHES the solver's reported h_k.
        n3 = 3
        Np = 4
        probs = (p = rand(rng, Np) .+ 0.2; p ./ sum(p))
        Yp = WeightedSpace(probs)
        Q = [let Mt = randn(rng, n3, n3); (Mt'Mt) ./ n3 .* 0.05 end for _ in 1:Np]
        F = QuadraticMap(Q, randn(rng, Np, n3), randn(rng, Np))
        f = TikhonovObjective(0.2)
        A = ConvexCombinationEnvelope(0.75, 0.9, probs)
        prob = RiskProblem(f, F, ZeroProx{Float64}(), A; Y=Yp)
        dsg = DualSPGOptions{Float64}(tol=1e-8, maxit=400)
        J0 = objective_value!(zeros(Np), prob, zeros(n3))
        res = solve(prob, zeros(n3); opts=TROptions{Float64}(maxit=300, dualspg=dsg))
        @test res.J < J0                                   # objective decreased
        @test res.hk < 1e-4                                # substantial stationarity progress
        @test isapprox(indep_hstat_w(prob, res.x, 1.0), res.hk; rtol=1e-2)  # metric h_k is consistent
    end
end
