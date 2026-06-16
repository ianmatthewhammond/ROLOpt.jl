# Phase 1, verification step 1: finite-difference derivative checks on f and F,
# adjoint (transpose) consistency, and Hessian symmetry. Mirrors ROL's
# Objective::checkGradient / checkHessVec / Constraint::checkAdjointConsistency.

@testset "derivative checks" begin
    rng = MersenneTwister(1)
    n = 6
    x = randn(rng, n); v = randn(rng, n); u = randn(rng, n)

    @testset "smooth objectives" begin
        for f in (DiagonalQuadratic(rand(rng, n) .+ 0.5, randn(rng, n)),
                  QuadraticObjective(let M = randn(rng, n, n); M'M + I end, randn(rng, n)),
                  TikhonovObjective(1.3))
            @test check_gradient(f, x, v) < 1e-9
            @test check_hessvec(f, x, v) < 1e-9
            @test check_hess_sym(f, x, u, v) < 1e-10
        end
    end

    @testset "vector maps + adjoint" begin
        m = 4
        Amat = randn(rng, m, n); c = randn(rng, m)
        F_aff = AffineMap(Amat, c)
        Q = [let M = randn(rng, n, n); M'M end for _ in 1:m]
        amat = randn(rng, m, n); cq = randn(rng, m)
        F_quad = QuadraticMap(Q, amat, cq)
        theta = randn(rng, m)
        for F in (F_aff, F_quad)
            @test check_jacobian(F, x, v) < 1e-8
            @test check_adjoint(F, x, v, theta) < 1e-10
        end
    end
end
