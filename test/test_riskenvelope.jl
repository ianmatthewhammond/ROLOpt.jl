# Risk envelopes under Euclidean and weighted (diagonal) Y metrics.

# Independent support function for the convex-combination envelope, via the
# metric-generic greedy water-filling solution of
#   sup{ Σ wᵢ θᵢ yᵢ : Σ pᵢ θᵢ = 1, l ≤ θᵢ ≤ u },  density = wᵢyᵢ/pᵢ.
function greedy_suppA(lambda, pq, probs, y, w)
    l = 1 - lambda; u = (1 - lambda) + lambda / (1 - pq)
    n = length(y)
    dens = [(w[i] * y[i]) / probs[i] for i in 1:n]
    ord = sortperm(dens; rev=true)
    theta = fill(l, n)
    budget = 1.0 - sum(probs .* l)
    for i in ord
        add = min(u - l, budget / probs[i])
        theta[i] = l + add
        budget -= probs[i] * add
        budget <= 0 && break
    end
    return sum(w .* theta .* y)
end

@testset "risk envelopes" begin
    rng = MersenneTwister(2)
    Xeuc = EuclideanSpace(Float64)

    @testset "constructor validation" begin
        @test_throws ArgumentError IntervalEnvelope(1.0, 0.0)            # a > b
        @test_throws ArgumentError IntervalEnvelope(0.0, 1.0, 0)         # n < 1
        @test_throws ArgumentError WeightedSpace([1.0, -0.5, 2.0])       # nonpositive weight
        @test_throws ArgumentError ConvexCombinationEnvelope(0.5, 0.9, [0.3, 0.3])   # sum != 1
        @test_throws ArgumentError ConvexCombinationEnvelope(0.5, 1.0, [0.5, 0.5])   # p >= 1
        @test_throws ArgumentError ConvexCombinationEnvelope(1.5, 0.9, [0.5, 0.5])   # lambda > 1
        @test_throws ArgumentError ConvexCombinationEnvelope(0.5, 0.9, [0.6, 0.6, -0.2])  # neg prob
    end

    @testset "IntervalEnvelope [0,1] scalar (Euclidean)" begin
        A = IntervalEnvelope(0.0, 1.0)
        for _ in 1:50
            y = [randn(rng)]
            @test suppA(A, Xeuc, y) ≈ max(0.0, y[1])
            @test projectdual(A, Xeuc, y) ≈ clamp.(y, 0.0, 1.0)
        end
        @test bound(A, Xeuc) == 1.0
        theta = [99.0]; initdual!(theta, A)
        @test 0.0 <= theta[1] <= 1.0
    end

    @testset "IntervalEnvelope box, weighted Y (support picks up weights)" begin
        a, b, n = -0.5, 2.0, 5
        A = IntervalEnvelope(a, b, n)
        w = rand(rng, n) .+ 0.3
        Yw = WeightedSpace(w)
        for _ in 1:50
            y = randn(rng, n)
            @test suppA(A, Yw, y) ≈ sum(w[i] * max(a * y[i], b * y[i]) for i in 1:n)
            @test projectdual(A, Yw, y) ≈ clamp.(y, a, b)   # box proj is metric-independent
        end
        @test bound(A, Yw) ≈ max(abs(a), abs(b)) * sqrt(sum(w))
    end

    @testset "ConvexCombinationEnvelope (E + CVaR)" begin
        n = 8
        probs = (p = rand(rng, n) .+ 0.1; p ./ sum(p))
        lambda, pq = 0.75, 0.9
        A = ConvexCombinationEnvelope(lambda, pq, probs)
        Yprob = WeightedSpace(probs)                 # natural Y = L²(P)

        @testset "support function is metric-generic (both Y metrics)" begin
            for _ in 1:30
                y = randn(rng, n)
                # probability metric: equals (1-λ)E + λ AVaR
                @test suppA(A, Yprob, y) ≈ greedy_suppA(lambda, pq, probs, y, probs) atol=1e-9
                # Euclidean metric: different functional, but self-consistent
                @test suppA(A, Xeuc, y) ≈ greedy_suppA(lambda, pq, probs, y, ones(n)) atol=1e-9
            end
            # the two metrics genuinely differ
            y = randn(rng, n)
            @test !isapprox(suppA(A, Yprob, y), suppA(A, Xeuc, y); atol=1e-3)
            # theta ≡ 1 is feasible
            theta = zeros(n); initdual!(theta, A)
            @test all(theta .== 1) && sum(probs .* theta) ≈ 1
        end

        @testset "projection feasibility + idempotence (probability metric)" begin
            for _ in 1:30
                y = 2 .* randn(rng, n) .+ 1
                th = projectdual(A, Yprob, y)
                @test sum(probs .* th) ≈ 1 atol=1e-9      # Σ pᵢ θᵢ = 1
                @test all(A.l - 1e-12 .<= th .<= A.u + 1e-12)
                th2 = projectdual(A, Yprob, th)
                @test th2 ≈ th atol=1e-9                  # idempotent
            end
        end

        @testset "projection variational inequality (y - θ̄, θ' - θ̄)_Y ≤ 0" begin
            y = 3 .* randn(rng, n)
            th = projectdual(A, Yprob, y)
            for _ in 1:200
                # random feasible θ' = projection of a random box point
                thp = projectdual(A, Yprob, A.l .+ (A.u - A.l) .* rand(rng, n))
                vi = inner(Yprob, y .- th, thp .- th)
                @test vi <= 1e-8
            end
        end

        @testset "metric-specific shift geometry (swap-bug guard)" begin
            y = 0.5 .* randn(rng, n) .+ 1     # keep many components interior
            inact(th) = findall(i -> A.l + 1e-6 < th[i] < A.u - 1e-6, 1:n)

            # Probability metric (w = probs): inactive shift θ̄ᵢ - yᵢ is UNIFORM.
            thp = projectdual(A, Yprob, y)
            I = inact(thp)
            @test length(I) >= 2
            d = [thp[i] - y[i] for i in I]
            @test maximum(d) - minimum(d) < 1e-7

            # Euclidean metric (w = 1) with the SAME Σ pᵢ θᵢ = 1 constraint:
            # inactive (θ̄ᵢ - yᵢ)/pᵢ is uniform (probability-weighted shift).
            the = projectdual(A, Xeuc, y)
            @test sum(probs .* the) ≈ 1 atol=1e-9
            J = inact(the)
            @test length(J) >= 2
            r = [(the[i] - y[i]) / probs[i] for i in J]
            @test maximum(r) - minimum(r) < 1e-7
            # The two projections genuinely differ (geometry depends on the metric).
            @test !isapprox(thp, the; atol=1e-3)
        end
    end

    @testset "SimplexEnvelope (probability simplex / hard max)" begin
        @testset "constructor validation" begin
            @test_throws ArgumentError SimplexEnvelope{Float64}(0)   # n < 1
        end

        @testset "support function is the max (Euclidean) / weighted max" begin
            n = 7
            A = SimplexEnvelope{Float64}(n)
            w = rand(rng, n) .+ 0.3
            Yw = WeightedSpace(w)
            for _ in 1:50
                y = randn(rng, n)
                @test suppA(A, Xeuc, y) ≈ maximum(y)                       # exact max
                @test suppA(A, Yw, y) ≈ maximum(w[i] * y[i] for i in 1:n)  # weighted max
            end
            @test bound(A, Xeuc) == 1.0
            @test bound(A, Yw) ≈ sqrt(maximum(w))                         # vertex norm
            theta = fill(99.0, n); initdual!(theta, A)
            @test all(theta .≈ 1 / n) && sum(theta) ≈ 1                   # uniform, feasible
        end

        @testset "projection feasibility + idempotence + VI (Euclidean & weighted)" begin
            n = 9
            A = SimplexEnvelope{Float64}(n)
            w = rand(rng, n) .+ 0.2
            for Y in (Xeuc, WeightedSpace(w))
                for _ in 1:40
                    y = 2 .* randn(rng, n) .+ 0.5
                    th = projectdual(A, Y, y)
                    @test sum(th) ≈ 1 atol=1e-9        # Σ θᵢ = 1 (NOT Σ pᵢ θᵢ)
                    @test all(th .>= -1e-12)           # θ ≥ 0
                    @test projectdual(A, Y, th) ≈ th atol=1e-9   # idempotent
                end
                # variational inequality (y - θ̄, θ' - θ̄)_Y ≤ 0 for feasible θ'
                y = 3 .* randn(rng, n)
                th = projectdual(A, Y, y)
                for _ in 1:100
                    thp = projectdual(A, Y, randn(rng, n))
                    @test inner(Y, y .- th, thp .- th) <= 1e-8
                end
            end
        end
    end
end
