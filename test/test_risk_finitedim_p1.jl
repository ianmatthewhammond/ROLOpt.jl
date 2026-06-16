# Phase 1, verification step 5: full Cauchy-only TR-Risk on a finite-dimensional
# analogue of the sparse-control problem (scalar Y, A=[0,1], BoxL1 phi, Tikhonov
# f, quadratic F). Convergence is the paper-style TARGET on these convex
# instances: drive h_k below 1e-8. Also asserts (i) an independent stationarity
# residual, (ii) the true objective evaluator uses exact F (not the
# linearization), and (iii) the F'' path is never invoked in Phase 1.

# Wrapper that counts F'' (applyAdjointHessian!) calls.
struct CountingMap{T,FM<:VectorMap{T}} <: VectorMap{T}
    F::FM
    nhess::Base.RefValue{Int}
end
CountingMap(F::VectorMap{T}) where {T} = CountingMap{T,typeof(F)}(F, Ref(0))
ROLOpt.rangedim(C::CountingMap) = rangedim(C.F)
ROLOpt.value!(b, C::CountingMap, x) = value!(b, C.F, x)
ROLOpt.applyJacobian!(o, C::CountingMap, v, x) = applyJacobian!(o, C.F, v, x)
ROLOpt.applyAdjointJacobian!(o, C::CountingMap, θ, x) = applyAdjointJacobian!(o, C.F, θ, x)
ROLOpt.applyAdjointHessian!(o, C::CountingMap, θ, v, x) =
    (C.nhess[] += 1; applyAdjointHessian!(o, C.F, θ, v, x))

# Independent stationarity residual h(x) = ||prox_{t0 psi_x}(x - t0 ∇f(x)) - x||/t0,
# built from scratch (fresh PsiModel) at the candidate x.
function indep_hstat(prob, x, t0)
    n = length(x)
    psi = PsiModel(prob.phi, prob.A, prob.F, prob.X, prob.Y, n)
    reanchor!(psi, x)
    g = gradient(prob.f, x)
    dg = similar(x); riesz!(dg, prob.X, g)
    z = x .- t0 .* dg
    p = similar(x); prox!(p, psi, z, t0)
    return spnorm(prob.X, p .- x) / t0
end

@testset "Phase 1 risk problem (sparse-control analogue)" begin
    rng = MersenneTwister(7)
    n = 6

    # Quadratic scalar map F(x) = ½ xᵀQx + aᵀx + c, Q small SPD.
    M = randn(rng, n, n); Q = (M'M) ./ n .* 0.2
    Fbase = QuadraticMap([Q], reshape(randn(rng, n), 1, n), [0.3])
    F = CountingMap(Fbase)

    f = TikhonovObjective(1e-1)
    phi = BoxL1Objective(fill(1e-2, n), fill(-5.0, n), fill(5.0, n))
    A = IntervalEnvelope(0.0, 1.0)
    prob = RiskProblem(f, F, phi, A)

    res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=2000))

    @testset "convergence target" begin
        @test res.converged
        @test res.hk <= 1e-8
        @test res.iters >= 1
        @test res.counters.nfval > 0 && res.counters.ngrad > 0 && res.counters.nprox > 0
    end

    @testset "independent stationarity residual" begin
        @test indep_hstat(prob, res.x, 1.0) < 1e-7
    end

    @testset "F'' path never invoked in Phase 1" begin
        @test F.nhess[] == 0
    end

    @testset "true objective uses exact F, not the linearization" begin
        x = randn(rng, n)
        Fbuf = zeros(1)
        Jtrue = objective_value!(Fbuf, prob, x)
        manual = value(f, x) + max(0.0, mapvalue(Fbase, x)[1]) + value(phi, x)
        @test Jtrue ≈ manual                      # objective_value! uses exact F

        # The model linearization ℓ_k(x) = F'(xk)(x-xk)+F(xk) genuinely differs
        # from the exact F(x) by the (nonzero) second-order term — confirming the
        # model and the true objective are distinct quantities.
        xk0 = zeros(n); xq = fill(1.0, n)         # deterministic, clear curvature gap
        ell = zeros(1)
        applyJacobian!(ell, Fbase, xq .- xk0, xk0)
        ell .+= mapvalue(Fbase, xk0)              # ℓ_k(xq)
        Ftrue = mapvalue(Fbase, xq)[1]
        @test !isapprox(ell[1], Ftrue; atol=1e-6)
    end
    # Note: weighted (non-Euclidean) X/Y spaces require a metric-aware phi.prox
    # and risk projection; deferred to Phase 2. Phase 1 validates Euclidean only.
end
