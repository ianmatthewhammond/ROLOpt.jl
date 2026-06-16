# Phase 1, verification step 4: with the risk term inactive (A = {0}), the
# algorithm reduces to ROL's TypeP LASSO test (diagonal-quadratic f + weighted/
# shifted L1). Compare to the closed-form `getSolution` from ROL test_01.cpp:
#   x_i = (clamp(a_i*y_i + b_i, -w_i, w_i) - b_i) / a_i.

@testset "reduce to ROL TypeP LASSO (A = {0})" begin
    rng = MersenneTwister(123)
    n = 5
    a = 5 .* rand(rng, n) .+ 1e-2      # a_i in (0, 5]
    b = 20 .* rand(rng, n) .- 10       # b_i in [-10, 10]
    w = rand(rng, n)                   # L1 weights in [0,1)
    y = 10 .* rand(rng, n) .- 5        # L1 shift

    f = DiagonalQuadratic(a, b)
    F = AffineMap(zeros(1, n), [0.0])  # F ≡ 0  => risk term inert
    phi = L1Objective(w, y)
    A = IntervalEnvelope(0.0, 0.0)     # {0}
    prob = RiskProblem(f, F, phi, A)

    res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=5000))
    xstar = @. (clamp(a * y + b, -w, w) - b) / a

    @test res.converged
    @test res.hk <= 1e-8
    @test norm(res.x - xstar) < 1e3 * sqrt(eps())
end
