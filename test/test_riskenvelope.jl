# Phase 1, verification step 2: IntervalEnvelope support function and projection
# against closed forms.

@testset "risk envelope (IntervalEnvelope)" begin
    rng = MersenneTwister(2)

    @testset "[0,1] scalar: suppA = (y)_+, projA = clamp" begin
        A = IntervalEnvelope(0.0, 1.0)
        for _ in 1:50
            y = [randn(rng)]
            @test suppA(A, y) ≈ max(0.0, y[1])
            @test projectdual(A, y) ≈ clamp.(y, 0.0, 1.0)
        end
        @test bound(A) == 1.0
        # initial dual must be feasible
        θ = [99.0]; initdual!(θ, A)
        @test 0.0 <= θ[1] <= 1.0
    end

    @testset "[a,b]^n box: suppA = Σ max(a yᵢ, b yᵢ)" begin
        a, b, n = -0.5, 2.0, 5
        A = IntervalEnvelope(a, b, n)
        for _ in 1:50
            y = randn(rng, n)
            @test suppA(A, y) ≈ sum(max(a * yi, b * yi) for yi in y)
            @test projectdual(A, y) ≈ clamp.(y, a, b)
        end
        @test bound(A) ≈ sqrt(n) * max(abs(a), abs(b))
    end
end
