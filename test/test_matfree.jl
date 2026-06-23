# Phase 1: user-selected F representations. The user picks how F is stored by the
# VectorMap type they pass — explicit AffineMap/QuadraticMap, pure matrix-free
# OperatorMap, or assemble-once-per-anchor CachedJacobianMap — and ROLOpt never
# infers/assembles A_k itself. The central claim: representations of the SAME map
# produce the same solver trajectory. Plus the prepare! anchor hook, the cache
# contract (value! must not corrupt the x_k linearization), VJP validation, a
# user factorization-cache pattern, curvature, and precision genericity.

# Matrix-free / cached builders for a LINEAR map F(x) = A x + c.
opmap_lin(::Type{T}, A, c) where {T} = OperatorMap{T}(; m=size(A, 1),
    value=(b, x) -> (mul!(b, A, x); b .+= c; b),
    jvp=(o, v, x) -> mul!(o, A, v),
    vjp=(o, th, x) -> mul!(o, transpose(A), th))
cj_lin(::Type{T}, A, c) where {T} = CachedJacobianMap{T}(; m=size(A, 1), Ak=zeros(T, size(A)...),
    value=(b, x) -> (mul!(b, A, x); b .+= c; b),
    assemble=(Ak, x) -> (copyto!(Ak, A); nothing))

@testset "matrix-free & cached-Jacobian F" begin

    @testset "(1) three-mode equivalence — linear F" begin
        rng = MersenneTwister(11); n = 4; m = 3
        A = randn(rng, m, n); c = randn(rng, m)
        f = TikhonovObjective(1.0); phi = L1Objective(fill(5e-2, n)); Arisk = IntervalEnvelope(0.0, 1.0)
        for sub in (:cauchy, :spg, :ncg)
            o = TROptions{Float64}(maxit=2000, subsolver=sub, matrix_free_reuse=:off)  # reuse :on tested in (9)
            re = solve(RiskProblem(f, AffineMap(A, c), phi, Arisk), zeros(n); opts=o)
            rb = solve(RiskProblem(f, opmap_lin(Float64, A, c), phi, Arisk), zeros(n); opts=o)
            rc = solve(RiskProblem(f, cj_lin(Float64, A, c), phi, Arisk), zeros(n); opts=o)
            @test rb.x == re.x && rb.J == re.J && rb.iters == re.iters    # bit-identical
            @test rc.converged == re.converged && norm(rc.x - re.x) < 1e-12
        end
    end

    @testset "(1b) three-mode equivalence — nonlinear F (QuadraticMap)" begin
        # A well-conditioned (Tikhonov + box risk) nonlinear problem: A_k = F'(x_k)
        # varies with the anchor, so CachedJacobianMap genuinely rebuilds it. (We avoid
        # the borderline CVaR-near-floor instance, where the cached map's independent
        # assembly — summed differently from QM's on-the-fly apply — would shift the
        # ~1e-8 stationarity floor and change the iteration count.)
        rng = MersenneTwister(12); n = 3; N = 4
        Q = [(M = randn(rng, n, n); (M'M) ./ n .* 0.05) for _ in 1:N]
        a = randn(rng, N, n); c = randn(rng, N)
        QM = QuadraticMap(Q, a, c)
        opQ = OperatorMap{Float64}(; m=N,                       # wraps QM's own actions
            value=(b, x) -> value!(b, QM, x),
            jvp=(o, v, x) -> applyJacobian!(o, QM, v, x),
            vjp=(o, th, x) -> applyAdjointJacobian!(o, QM, th, x))
        cjQ = CachedJacobianMap{Float64}(; m=N, Ak=zeros(N, n),  # rebuilds A_k=F'(x_k) per anchor
            value=(b, x) -> value!(b, QM, x),
            assemble=(Ak, x) -> (for j in 1:N
                @views Ak[j, :] .= Q[j] * x .+ a[j, :]
            end; nothing))
        f = TikhonovObjective(1.0); phi = L1Objective(fill(5e-2, n)); Arisk = IntervalEnvelope(0.0, 1.0)
        for sub in (:cauchy, :spg, :ncg)
            o = TROptions{Float64}(maxit=3000, subsolver=sub, matrix_free_reuse=:off)  # reuse :on tested in (9)
            re = solve(RiskProblem(f, QM, phi, Arisk), zeros(n); opts=o)
            rb = solve(RiskProblem(f, opQ, phi, Arisk), zeros(n); opts=o)
            rc = solve(RiskProblem(f, cjQ, phi, Arisk), zeros(n); opts=o)
            @test re.converged && rb.converged && rc.converged
            @test norm(rb.x - re.x) < 1e-12     # opmap wraps QM actions → identical arithmetic
            @test norm(rc.x - re.x) < 1e-6      # cached: independent assembly, same minimizer
        end
    end

    @testset "(2) weighted X/Y metric" begin
        rng = MersenneTwister(13); n = 4; m = 3
        A = randn(rng, m, n); c = randn(rng, m)
        Xw = WeightedSpace(rand(rng, n) .+ 0.5); Yw = WeightedSpace(rand(rng, m) .+ 0.5)
        f = TikhonovObjective(1.0); phi = L1Objective(fill(5e-2, n)); Arisk = IntervalEnvelope(0.0, 1.0)
        o = TROptions{Float64}(maxit=2000, subsolver=:spg, matrix_free_reuse=:off)  # reuse :on tested in (9)
        re = solve(RiskProblem(f, AffineMap(A, c), phi, Arisk; X=Xw, Y=Yw), zeros(n); opts=o)
        rb = solve(RiskProblem(f, opmap_lin(Float64, A, c), phi, Arisk; X=Xw, Y=Yw), zeros(n); opts=o)
        rc = solve(RiskProblem(f, cj_lin(Float64, A, c), phi, Arisk; X=Xw, Y=Yw), zeros(n); opts=o)
        @test rb.x == re.x                          # metric sandwich is the framework's job
        @test norm(rc.x - re.x) < 1e-12
    end

    @testset "(3) wrong-VJP detection (check_adjoint + validate_F)" begin
        rng = MersenneTwister(14); n = 4; m = 3
        A = randn(rng, m, n); c = randn(rng, m)
        good = opmap_lin(Float64, A, c)
        bad = OperatorMap{Float64}(; m=m,           # VJP drops the transpose → inconsistent
            value=(b, x) -> (mul!(b, A, x); b .+= c; b),
            jvp=(o, v, x) -> mul!(o, A, v),
            vjp=(o, th, x) -> (o .= 0; o))
        x = randn(rng, n); v = randn(rng, n); th = randn(rng, m)
        @test check_adjoint(good, x, v, th) < 1e-10
        @test check_adjoint(bad, x, v, th) > 1e-2
        f = TikhonovObjective(1.0); phi = ZeroProx{Float64}(); Arisk = IntervalEnvelope(0.0, 1.0)
        @test solve(RiskProblem(f, good, phi, Arisk), zeros(n);
                    opts=TROptions{Float64}(validate_F=true)) isa TRResult
        @test_throws ArgumentError solve(RiskProblem(f, bad, phi, Arisk), zeros(n);
                                         opts=TROptions{Float64}(validate_F=true))
    end

    @testset "(4) prepare! discipline (once per anchor, not per apply)" begin
        rng = MersenneTwister(15); n = 4; m = 3
        A = randn(rng, m, n); c = randn(rng, m)
        pc = Ref(0); applies = Ref(0)
        F = OperatorMap{Float64}(; m=m,
            value=(b, x) -> (mul!(b, A, x); b .+= c; b),
            jvp=(o, v, x) -> (applies[] += 1; mul!(o, A, v)),
            vjp=(o, th, x) -> (applies[] += 1; mul!(o, transpose(A), th)),
            prepare=(x) -> (pc[] += 1; nothing))
        f = TikhonovObjective(1.0); phi = L1Objective(fill(5e-2, n)); Arisk = IntervalEnvelope(0.0, 1.0)
        res = solve(RiskProblem(f, F, phi, Arisk), zeros(n); opts=TROptions{Float64}(maxit=2000))
        @test res.converged
        @test 1 <= pc[] <= res.iters + 1            # initial anchor + accepted steps (exact path)
        @test applies[] > 10 * pc[]                  # applies ≫ prepares ⇒ per-anchor, not per-apply
    end

    @testset "(5) cache contract: value! must not corrupt the x_k Jacobian" begin
        rng = MersenneTwister(16); n = 3; N = 4
        Q = [(M = randn(rng, n, n); (M'M) ./ n .* 0.05) for _ in 1:N]
        a = randn(rng, N, n); c = randn(rng, N)
        QM = QuadraticMap(Q, a, c)
        assemble = (Ak, x) -> (for j in 1:N
            @views Ak[j, :] .= Q[j] * x .+ a[j, :]
        end; nothing)
        side = zeros(n)
        cjPlain = CachedJacobianMap{Float64}(; m=N, Ak=zeros(N, n),
            value=(b, x) -> (value!(b, QM, x); b), assemble=assemble)
        cjAdv = CachedJacobianMap{Float64}(; m=N, Ak=zeros(N, n),       # value! also writes a SEPARATE cache
            value=(b, x) -> (side .= 2 .* x; value!(b, QM, x); b), assemble=assemble)
        # Unit check (the core contract): a value! at x ≠ anchor leaves the assembled
        # A_k untouched — value! mutates only its own `side` cache, never the linearization.
        prepare!(cjAdv, zeros(n)); Ak0 = copy(cjAdv.Ak)
        value!(zeros(N), cjAdv, ones(n))
        @test cjAdv.Ak == Ak0
        # End-to-end with a tiny initial radius (forces early rejected trials): a value!
        # carrying its own separate state yields a bit-identical solve vs the plain map,
        # i.e. the value!-at-rejected-trial calls never perturbed the x_k Jacobian.
        f = TikhonovObjective(1.0); phi = L1Objective(fill(5e-2, n)); Arisk = IntervalEnvelope(0.0, 1.0)
        o = TROptions{Float64}(maxit=500, subsolver=:spg, delta0=1e-2, gamma3=2.0)
        rp = solve(RiskProblem(f, cjPlain, phi, Arisk), zeros(n); opts=o)
        ra = solve(RiskProblem(f, cjAdv, phi, Arisk), zeros(n); opts=o)
        @test ra.x == rp.x && ra.iters == rp.iters
    end

    @testset "(6) user factorization cache (LU once per anchor, back-subs)" begin
        rng = MersenneTwister(17); n = 4; m = 3
        B = randn(rng, m, n)
        K = Matrix(0.5I, m, m) + (R = randn(rng, m, m); 0.1 * (R + R') / 2)   # SPD, well-conditioned
        # F(x) = (K \ B) x : a reduced map whose JVP/VJP are K- and Kᵀ-solves. The user
        # factorizes K once per anchor in prepare! and reuses the factors as back-subs.
        Fref = K \ B                                # explicit reduced Jacobian (the oracle)
        facref = Ref{Any}(nothing); nfac = Ref(0)
        Bv = zeros(m); xtmp = zeros(m)
        F = OperatorMap{Float64}(; m=m,
            value=(b, x) -> (mul!(Bv, B, x); ldiv!(b, facref[], Bv); b),
            jvp=(o, v, x) -> (mul!(Bv, B, v); ldiv!(o, facref[], Bv); o),
            vjp=(o, th, x) -> (ldiv!(xtmp, transpose(facref[]), th); mul!(o, transpose(B), xtmp); o),
            prepare=(x) -> (nfac[] += 1; facref[] = lu(K); nothing))
        f = TikhonovObjective(1.0); phi = L1Objective(fill(5e-2, n)); Arisk = IntervalEnvelope(0.0, 1.0)
        o = TROptions{Float64}(maxit=2000, subsolver=:spg)
        re = solve(RiskProblem(f, AffineMap(Fref, zeros(m)), phi, Arisk), zeros(n); opts=o)
        rb = solve(RiskProblem(f, F, phi, Arisk), zeros(n); opts=o)
        @test rb.converged && norm(rb.x - re.x) < 1e-8
        @test 1 <= nfac[] <= rb.iters + 1            # factorized per anchor, not per apply
    end

    @testset "(7) curvature: ahp equivalence and missing-ahp error" begin
        rng = MersenneTwister(18); n = 3; N = 4
        Q = [(M = randn(rng, n, n); (M'M) ./ n .* 0.05) for _ in 1:N]
        a = randn(rng, N, n); c = randn(rng, N)
        QM = QuadraticMap(Q, a, c)
        opH = OperatorMap{Float64}(; m=N,
            value=(b, x) -> value!(b, QM, x),
            jvp=(o, v, x) -> applyJacobian!(o, QM, v, x),
            vjp=(o, th, x) -> applyAdjointJacobian!(o, QM, th, x),
            ahp=(o, th, v, x) -> applyAdjointHessian!(o, QM, th, v, x))
        opNo = OperatorMap{Float64}(; m=N,           # same, but no ahp
            value=(b, x) -> value!(b, QM, x),
            jvp=(o, v, x) -> applyJacobian!(o, QM, v, x),
            vjp=(o, th, x) -> applyAdjointJacobian!(o, QM, th, x))
        f = TikhonovObjective(0.2); phi = ZeroProx{Float64}()
        probs = (p = rand(MersenneTwister(5), N) .+ 0.2; p ./ sum(p)); Yp = WeightedSpace(probs)
        Arisk = ConvexCombinationEnvelope(0.75, 0.9, probs)
        o = TROptions{Float64}(maxit=300, subsolver=:spg, use_curvature=true, matrix_free_reuse=:off)
        re = solve(RiskProblem(f, QM, phi, Arisk; Y=Yp), zeros(n); opts=o)
        rb = solve(RiskProblem(f, opH, phi, Arisk; Y=Yp), zeros(n); opts=o)
        @test norm(rb.x - re.x) < 1e-10              # ahp wraps QM's F'' → identical (reuse off)
        @test_throws ArgumentError solve(RiskProblem(f, opNo, phi, Arisk; Y=Yp), zeros(n); opts=o)
    end

    @testset "(8b) CachedJacobianMap dimension validation" begin
        val = (b, x) -> (b .= 0; b)
        asm = (Ak, x) -> nothing
        @test_throws ArgumentError CachedJacobianMap{Float64}(;        # rows ≠ m
            m=2, Ak=zeros(3, 4), value=val, assemble=asm)
        F = CachedJacobianMap{Float64}(; m=3, Ak=zeros(3, 4), value=val,
            assemble=(Ak, x) -> zeros(2, 4))                          # returns wrong shape
        @test_throws ArgumentError prepare!(F, zeros(4))
    end

    @testset "(8) precision genericity (Float32 / BigFloat)" begin
        n = 3
        for (T, tol) in ((Float32, 1.0f-5), (BigFloat, BigFloat(1e-20)))
            A = ones(T, 1, n); c = T[0.5]
            prob_e = RiskProblem(TikhonovObjective(one(T)), AffineMap(A, c), ZeroProx{T}(), IntervalEnvelope(zero(T), one(T)))
            prob_b = RiskProblem(TikhonovObjective(one(T)), opmap_lin(T, A, c), ZeroProx{T}(), IntervalEnvelope(zero(T), one(T)))
            opts = T == Float32 ? TROptions{T}(htol=1.0f-3, maxit=200, dualspg=DualSPGOptions{T}(tol=1.0f-5)) :
                                  TROptions{T}(htol=T(1e-20), maxit=300, dualspg=DualSPGOptions{T}(tol=T(1e-30), gmin=T(1e-40)))
            re = solve(prob_e, zeros(T, n); opts=opts)
            rb = solve(prob_b, zeros(T, n); opts=opts)
            @test rb.x == re.x                       # identical arithmetic at any precision
        end
    end

    @testset "(9) matrix-free line-search reuse: matches naive, far fewer applies" begin
        # reuse :on must agree with :off (the AXPY of the linear adjoint/forward
        # line-search pieces equals the recompute up to rounding) while doing far fewer
        # operator applies — each a PDE solve in a real matrix-free F. :spg exercises
        # the dual-SPG adjoint reuse; :ncg also the dbls! forward-JVP reuse. Counters
        # live in the OperatorMap closures.
        jc = Ref(0); vc = Ref(0)

        @testset "well-conditioned: converges both modes, tight agreement, >2× fewer VJPs" begin
            rng = MersenneTwister(7); n = 5; m = 4
            A = randn(rng, m, n); c = randn(rng, m)
            opF = OperatorMap{Float64}(; m=m, value=(b, x) -> (mul!(b, A, x); b .+= c; b),
                jvp=(o, v, x) -> (jc[] += 1; mul!(o, A, v)),
                vjp=(o, th, x) -> (vc[] += 1; mul!(o, transpose(A), th)))
            prob = RiskProblem(TikhonovObjective(1.0), opF, L1Objective(fill(2e-2, n)), IntervalEnvelope(0.0, 1.0))
            for sub in (:spg, :ncg)
                cnts = Dict{Symbol,NTuple{2,Int}}(); xs = Dict{Symbol,Vector{Float64}}()
                for mode in (:off, :on)
                    jc[] = 0; vc[] = 0
                    r = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=2000, subsolver=sub, matrix_free_reuse=mode))
                    @test r.converged
                    cnts[mode] = (jc[], vc[]); xs[mode] = copy(r.x)
                end
                @test norm(xs[:on] - xs[:off]) < 1e-9       # AXPY ≈ recompute (well-conditioned ⇒ tight)
                @test 2 * cnts[:on][2] < cnts[:off][2]      # VJP (adjoint) applies: >2× fewer
                sub === :ncg && @test cnts[:on][1] < cnts[:off][1]   # dbls! also cuts forward JVPs
            end
        end

        @testset "harder instance (CVaR :ncg): dramatic adjoint-apply reduction" begin
            srng = MersenneTwister(5); n = 3; N = 4
            probs = (p = rand(srng, N) .+ 0.2; p ./ sum(p)); Yp = WeightedSpace(probs)
            Q = [(M = randn(srng, n, n); (M'M) ./ n .* 0.05) for _ in 1:N]
            a = randn(srng, N, n); c = randn(srng, N); QM = QuadraticMap(Q, a, c)
            opQ = OperatorMap{Float64}(; m=N, value=(b, x) -> value!(b, QM, x),
                jvp=(o, v, x) -> (jc[] += 1; applyJacobian!(o, QM, v, x)),
                vjp=(o, th, x) -> (vc[] += 1; applyAdjointJacobian!(o, QM, th, x)))
            prob = RiskProblem(TikhonovObjective(0.2), opQ, ZeroProx{Float64}(),
                               ConvexCombinationEnvelope(0.75, 0.9, probs); Y=Yp)
            cnts = Dict{Symbol,NTuple{2,Int}}(); xs = Dict{Symbol,Vector{Float64}}()
            for mode in (:off, :on)
                jc[] = 0; vc[] = 0
                r = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=200, subsolver=:ncg, matrix_free_reuse=mode))
                @test r.converged
                cnts[mode] = (jc[], vc[]); xs[mode] = copy(r.x)
            end
            @test norm(xs[:on] - xs[:off]) < 1e-6
            @test 10 * cnts[:on][2] < cnts[:off][2]         # >10× fewer adjoint applies
        end
    end
end
