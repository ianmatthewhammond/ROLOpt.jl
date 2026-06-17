# Phase 1, verification step 3: Algorithm 2 (prox of psi_k via dual SPG).
# No QP solver is used (stdlib only). We check against:
#   (a) an independent low-dim dual maximizer we write here (golden section for a
#       scalar envelope; dense grid for a 2-D envelope),
#   (b) the primal-dual optimality residual: theta_bar is a fixed point of the
#       projected ascent, and p = prox_{t phi}(z - t A* theta_bar).

# Independent dual objective d(theta) for an AffineMap F (Euclidean spaces):
#   p(theta) = prox_{t phi}(z - t A' theta)
#   d(theta) = 1/(2t)||p-z||^2 + phi(p) + theta . (A (p - xk) + b_k)
function _indep_dual(F::AffineMap, phi, xk, z, t, theta)
    bk = F.A * xk + F.c
    p = prox(phi, z - t * (F.A' * theta), t)
    ell = F.A * (p - xk) + bk
    d = dot(p - z, p - z) / (2t) + value(phi, p) + dot(theta, ell)
    return d, p, ell
end

@testset "Algorithm 2: prox of psi_k (dual SPG)" begin
    rng = MersenneTwister(3)
    X = EuclideanSpace(Float64); Y = EuclideanSpace(Float64)

    @testset "scalar envelope vs golden-section dual" begin
        n = 4
        for trial in 1:8
            F = AffineMap(randn(rng, 1, n), randn(rng, 1))
            phi = L1Objective(rand(rng, n) .+ 0.1)
            A = IntervalEnvelope(0.0, 1.0)
            psi = PsiModel(phi, A, F, X, Y, n)
            xk = randn(rng, n); reanchor!(psi, xk)
            z = randn(rng, n); t = 0.3 + rand(rng)

            out = similar(z)
            res = prox!(out, psi, z, t)
            @test res.converged

            # Independent 1-D maximizer of d over [0,1] (golden section).
            obj(theta1) = _indep_dual(F, phi, xk, z, t, [theta1])[1]
            lo, hi = 0.0, 1.0
            gr = (sqrt(5) - 1) / 2
            c1 = hi - gr * (hi - lo); c2 = lo + gr * (hi - lo)
            f1 = obj(c1); f2 = obj(c2)
            for _ in 1:200
                if f1 < f2
                    lo = c1; c1 = c2; f1 = f2
                    c2 = lo + gr * (hi - lo); f2 = obj(c2)
                else
                    hi = c2; c2 = c1; f2 = f1
                    c1 = hi - gr * (hi - lo); f1 = obj(c1)
                end
            end
            thetastar = (lo + hi) / 2
            _, pstar, _ = _indep_dual(F, phi, xk, z, t, [thetastar])

            @test isapprox(psi.theta[1], thetastar; atol=1e-5)
            @test isapprox(out, pstar; atol=1e-6)

            # Primal-dual residual: theta_bar fixed under projected ascent.
            _, p, ell = _indep_dual(F, phi, xk, z, t, psi.theta)
            @test isapprox(out, p; atol=1e-8)
            resid = clamp(psi.theta[1] + ell[1], 0.0, 1.0) - psi.theta[1]
            @test abs(resid) < 1e-6
        end
    end

    @testset "2-D box envelope vs dense grid" begin
        n = 3
        F = AffineMap(randn(rng, 2, n), randn(rng, 2))
        phi = L1Objective(rand(rng, n) .+ 0.1)
        A = IntervalEnvelope(0.0, 1.0, 2)
        psi = PsiModel(phi, A, F, X, Y, n)
        xk = randn(rng, n); reanchor!(psi, xk)
        z = randn(rng, n); t = 0.5
        out = similar(z); res = prox!(out, psi, z, t)
        @test res.converged

        # Independent dense-grid maximizer over [0,1]^2.
        best = -Inf; thetabest = [0.0, 0.0]
        gridvals = range(0.0, 1.0; length=151)
        for t1 in gridvals, t2 in gridvals
            d, _, _ = _indep_dual(F, phi, xk, z, t, [t1, t2])
            if d > best; best = d; thetabest = [t1, t2]; end
        end
        @test isapprox(psi.theta, thetabest; atol=2e-2)   # grid resolution ~1/150
        _, pbest, _ = _indep_dual(F, phi, xk, z, t, psi.theta)
        @test isapprox(out, pbest; atol=1e-8)
    end

    @testset "inactive envelope A={0} reduces to prox_phi" begin
        n = 4
        F = AffineMap(randn(rng, 1, n), randn(rng, 1))
        phi = L1Objective(rand(rng, n) .+ 0.1)
        A = IntervalEnvelope(0.0, 0.0)         # {0}
        psi = PsiModel(phi, A, F, X, Y, n)
        reanchor!(psi, randn(rng, n))
        z = randn(rng, n); t = 0.7
        out = similar(z); prox!(out, psi, z, t)
        @test isapprox(out, prox(phi, z, t); atol=1e-12)
    end

    @testset "prox! leaves the dual in psi.theta; snapshot needs a copy" begin
        # prox! now returns status only (isbits DualSPGResult). The optimal dual is
        # left in psi.theta and OVERWRITTEN by the next prox!, so a caller that needs
        # to retain it must copy first. With ZeroProx and A = [0,1], p(theta) =
        # z - theta*ones and grad d(theta) = sum(z) - n*theta, so the dual maximizer
        # is theta* = clamp(sum(z)/n, 0, 1) — analytically predictable.
        n = 3
        F = AffineMap(ones(1, n), [0.0])
        psi = PsiModel(ZeroProx{Float64}(), IntervalEnvelope(0.0, 1.0), F, X, Y, n)
        reanchor!(psi, zeros(n))
        out = zeros(n)
        prox!(out, psi, fill(-5.0 / n, n), 1.0)         # sum(z) = -5  => theta* = 0
        @test isapprox(psi.theta[1], 0.0; atol=1e-8)
        snap = copy(psi.theta)                          # snapshot BEFORE the next prox!
        prox!(out, psi, fill(5.0, n), 1.0)              # sum(z) = 5n  => theta* = 1
        @test isapprox(psi.theta[1], 1.0; atol=1e-8)    # psi.theta overwritten in place
        @test isapprox(snap[1], 0.0; atol=1e-8)         # the copy retained the first dual
    end

    @testset "DualSPGOptions requires memory >= 1" begin
        # Algorithm 2's nonmonotone reference needs a positive window; the workspace
        # ring would otherwise be empty (mismatch with opts.memory).
        @test_throws ArgumentError DualSPGOptions{Float64}(memory=0)
        @test_throws ArgumentError DualSPGOptions{Float64}(memory=-1)
        @test DualSPGOptions{Float64}(memory=1).memory == 1
        @test DualSPGOptions{Float64}(memory=5).memory == 5
    end
end
