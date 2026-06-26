# SimplexEnvelope end-to-end: ROLOpt minimizes a true (sampled) max, and the
# moving_sample reset-on-reanchor hook. The analytic instance:
#   min_x  max_i x_i + (tau/2)‖x‖²     (F = identity, A = simplex, phi = 0)
# By symmetry the optimum is x* = -1/(n·tau)·1 (dual theta uniform), so
#   x*_i = -1/(n·tau),  J* = -1/(2·n·tau).

@testset "SimplexEnvelope true-max solve" begin
    n = 5; tau = 1.0
    Aid = zeros(n, n); for i in 1:n; Aid[i, i] = 1.0; end   # F(x) = I·x + 0
    F = AffineMap(Aid, zeros(n))
    f = TikhonovObjective(tau)
    phi = ZeroProx()
    A = SimplexEnvelope{Float64}(n)
    prob = RiskProblem(f, F, phi, A)

    res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=2000))
    xstar = fill(-1 / (n * tau), n)
    Jstar = -1 / (2 * n * tau)

    @test res.converged
    @test res.hk <= 1e-8
    @test res.x ≈ xstar atol=1e-6
    @test res.J ≈ Jstar atol=1e-8

    # The objective evaluator uses the exact max, not a per-component thing.
    Fbuf = zeros(n)
    @test objective_value!(Fbuf, prob, res.x) ≈
          value(f, res.x) + maximum(mapvalue(F, res.x)) + value(phi, res.x)
end

@testset "moving_sample reset-on-reanchor" begin
    n = 5; tau = 1.0
    A = SimplexEnvelope{Float64}(n)
    phi = ZeroProx()
    X = EuclideanSpace(Float64); Y = EuclideanSpace(Float64)

    # Identity map, once as a fixed AffineMap and once as a moving-sample OperatorMap.
    Aid = zeros(n, n); for i in 1:n; Aid[i, i] = 1.0; end
    Ffixed = AffineMap(Aid, zeros(n))
    Fmoving = OperatorMap{Float64}(; m=n,
        value = (b, x) -> copyto!(b, x),
        jvp   = (o, v, x) -> copyto!(o, v),
        vjp   = (o, th, x) -> copyto!(o, th),   # F'ᵀ = I (bare transpose)
        moving_sample = true)

    @test moving_sample(Fmoving) == true
    @test moving_sample(Ffixed) == false

    # reanchor! re-seeds theta via warmdual! (argmax vertex for the simplex) ONLY for
    # the moving-sample map. F is the identity, so b_k = x_anchor ⇒ θ = e_argmax(x_anchor).
    psim = PsiModel(phi, A, Fmoving, X, Y, n)
    fill!(psim.theta, 7.0)
    xa = randn(n)
    reanchor!(psim, xa)
    @test psim.theta ≈ [i == argmax(xa) ? 1.0 : 0.0 for i in 1:n]   # warm dual = max vertex

    psif = PsiModel(phi, A, Ffixed, X, Y, n)
    fill!(psif.theta, 7.0)
    reanchor!(psif, randn(n))
    @test psif.theta ≈ fill(7.0, n)            # warm start preserved (no reset)

    # And the reset does not break convergence: the moving-sample identity map still
    # reaches the analytic optimum x* = -1/(n·tau).
    f = TikhonovObjective(tau)
    prob = RiskProblem(f, Fmoving, phi, A; X=X, Y=Y)
    res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=2000))
    @test res.converged
    @test res.x ≈ fill(-1 / (n * tau), n) atol=1e-6
end
