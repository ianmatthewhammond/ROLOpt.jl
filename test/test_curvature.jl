# Phase 2b: the (theta_k, F''(x_k) ·) curvature term in B_k, with theta_k frozen
# from the once-per-iteration stationarity prox. Opt-in via use_curvature; default
# (false) keeps the f-only model and never touches the F'' path.

# Counting wrapper to confirm whether the F'' path is invoked.
struct CountHessMap{T,FM<:VectorMap{T}} <: VectorMap{T}
    F::FM
    nh::Base.RefValue{Int}
end
CountHessMap(F::VectorMap{T}) where {T} = CountHessMap{T,typeof(F)}(F, Ref(0))
ROLOpt.rangedim(c::CountHessMap) = rangedim(c.F)
ROLOpt.value!(b, c::CountHessMap, x) = value!(b, c.F, x)
ROLOpt.applyJacobian!(o, c::CountHessMap, v, x) = applyJacobian!(o, c.F, v, x)
ROLOpt.applyAdjointJacobian!(o, c::CountHessMap, t, x) = applyAdjointJacobian!(o, c.F, t, x)
ROLOpt.applyAdjointHessian!(o, c::CountHessMap, t, v, x) =
    (c.nh[] += 1; applyAdjointHessian!(o, c.F, t, v, x))

@testset "F'' curvature term" begin
    rng = MersenneTwister(51)

    @testset "curvature converges to the convex optimum; F'' invoked only when enabled" begin
        # CVaR with quadratic F (nonzero F''). J is convex, so ron's h_k <= htol
        # certifies the global minimizer. On this instance the f-only model stalls
        # near the solution while the curvature model converges quickly — exactly
        # what the (theta_k, F'') term is for.
        n = 3; N = 4
        probs = (p = rand(rng, N) .+ 0.2; p ./ sum(p))
        Yp = WeightedSpace(probs)
        Q = [let Mt = randn(rng, n, n); (Mt'Mt) ./ n .* 0.05 end for _ in 1:N]
        Fbase = QuadraticMap(Q, randn(rng, N, n), randn(rng, N))
        f = TikhonovObjective(0.2)
        A = ConvexCombinationEnvelope(0.75, 0.9, probs)

        Foff = CountHessMap(Fbase)
        roff = solve(RiskProblem(f, Foff, ZeroProx{Float64}(), A; Y=Yp), zeros(n);
                     opts=TROptions{Float64}(maxit=400, subsolver=:spg, use_curvature=false))
        Fon = CountHessMap(Fbase)
        ron = solve(RiskProblem(f, Fon, ZeroProx{Float64}(), A; Y=Yp), zeros(n);
                    opts=TROptions{Float64}(maxit=400, subsolver=:spg, use_curvature=true))

        @test ron.converged && ron.iters < 100      # curvature certifies the optimum, fast
        @test Foff.nh[] == 0                          # f-only model never touches F''
        @test Fon.nh[]  > 0                           # curvature model does
        @test isapprox(roff.J, ron.J; atol=1e-6)     # both reach the same objective
    end

    @testset "B_k with curvature is symmetric" begin
        n = 4; m = 3
        Q = [let Mt = randn(rng, n, n); Mt'Mt end for _ in 1:m]
        F = QuadraticMap(Q, randn(rng, m, n), randn(rng, m))
        f = QuadraticObjective(let M = randn(rng, n, n); M'M + I end, randn(rng, n))
        X = EuclideanSpace(Float64)
        model = QuadraticModel(f, F, X, n, m)
        setmodel!(model, randn(rng, n), randn(rng, n))
        freeze_curvature!(model, randn(rng, m))
        for _ in 1:20
            u = randn(rng, n); v = randn(rng, n)
            Bu = modelhessvec!(zeros(n), model, u)
            Bv = modelhessvec!(zeros(n), model, v)
            @test apply(Bu, v) ≈ apply(Bv, u) atol=1e-9
        end
    end

    @testset "theta_k is frozen (independent of psi's live dual)" begin
        n = 4; m = 3
        Q = [let Mt = randn(rng, n, n); Mt'Mt .* 0.1 end for _ in 1:m]
        F = QuadraticMap(Q, randn(rng, m, n), randn(rng, m))
        f = TikhonovObjective(1.0)
        X = EuclideanSpace(Float64); Y = EuclideanSpace(Float64)
        model = QuadraticModel(f, F, X, n, m)
        setmodel!(model, randn(rng, n), randn(rng, n))
        psi = PsiModel(ZeroProx{Float64}(), IntervalEnvelope(0.0, 1.0), F, X, Y, n)
        reanchor!(psi, model.xk)

        freeze_curvature!(model, copy(psi.theta))
        v = randn(rng, n)
        r1 = copy(modelhessvec!(zeros(n), model, v))
        # Mutate psi's live dual (as the subproblem proxes would); B_k must not move.
        fill!(psi.theta, 0.9)
        prox!(zeros(n), psi, randn(rng, n), 0.5)         # also churns psi.theta
        r2 = modelhessvec!(zeros(n), model, v)
        @test r1 == r2                                    # frozen model Hessian
    end

    @testset "default model is f-only (no F'' fields touched)" begin
        n = 3
        f = TikhonovObjective(1.0)
        model = QuadraticModel(f, EuclideanSpace(Float64), n)
        @test model.F === nothing
        setmodel!(model, zeros(n), ones(n))
        @test modelhessvec!(zeros(n), model, ones(n)) == ones(n)   # ∇²f = I
    end
end
