# Phase 3a: type-stability (@inferred) and zero-allocation (@allocated) guards for
# the hot kernels. These DRIVE the 3a hardening and lock it in:
#   * dcauchy! used to infer Tuple{Float64,Any,Any,Any,Float64} (boxed closures);
#   * the dual-SPG prox allocated a dual copy + push!/popfirst! per call;
#   * the QuadraticMap adjoints allocated a reused `similar` temp per call;
#   * dcauchy!/dspg! allocated a Ref for the worst-residual accumulator.
# A box (Interval) envelope is used so suppA/project! are themselves alloc-free
# (CVaR's water-filling sort is a documented bounded exception — see riskenvelope.jl).
# The _solve_*/dncg! and precision (BigFloat/Float32) guards land in 3b/3d/3e.

# Allocation measured through a type-stable barrier: a bare `@allocated f(x)` in
# test/global scope counts the BOXING of an isbits/Float64 return under dynamic
# dispatch, not the kernel's real heap traffic. Specializing on the function and
# arg types removes that artifact.
nbytes(f::F, args::Vararg{Any,N}) where {F,N} = @allocated f(args...)

@testset "perf: type-stability + zero allocation (hot kernels)" begin
    rng = MersenneTwister(99)
    n = 4; m = 3
    f = TikhonovObjective(1.0)
    Q = [let M = randn(rng, n, n); (M'M) ./ n end for _ in 1:m]   # symmetric PSD
    F = QuadraticMap(Q, randn(rng, m, n), randn(rng, m))
    X = EuclideanSpace(Float64); Y = EuclideanSpace(Float64)
    phi = ZeroProx{Float64}()
    A = IntervalEnvelope(0.0, 1.0, m)
    psi = PsiModel(phi, A, F, X, Y, n)
    xk = randn(rng, n); reanchor!(psi, xk)

    modelf = QuadraticModel(f, X, n)               # f-only B_k
    setmodel!(modelf, xk, randn(rng, n))
    modelc = QuadraticModel(f, F, X, Y, n, m)       # curvature B_k (exercises F'' adjoint)
    setmodel!(modelc, xk, randn(rng, n))
    freeze_curvature!(modelc, ones(m))

    # InexactMap wrapping the QuadraticMap: its scratch-forwarding adjoint must keep
    # _dual_d! allocation-free (the generic fallback would drop the buffer and hit
    # the allocating 4-arg QuadraticMap method).
    imap = ROLOpt.InexactMap(F, n)
    imap.E .= 0.01 .* randn(rng, m, n); imap.eb .= 0.01 .* randn(rng, m)
    psi_im = PsiModel(phi, A, imap, X, Y, n); reanchor!(psi_im, xk)

    z = randn(rng, n); t = 0.5
    theta = ones(m); pbuf = zeros(n); mubuf = zeros(m)
    v = randn(rng, n); outn = zeros(n); stepn = zeros(n); dgn = randn(rng, n)
    cnt = Counters()
    cp = TROptions{Float64}().cauchy
    sp = SpgParams{Float64}()
    s0 = value(f, xk); n0 = value(psi, xk); dg = randn(rng, n)
    s = zeros(n); px = zeros(n); dwa = zeros(n); dwa1 = zeros(n)
    gmodc = zeros(n); gmodp = zeros(n); pgi = zeros(n); hs = zeros(n); dx = zeros(n)

    @testset "@inferred: concrete returns" begin
        @test (@inferred ROLOpt._dual_d!(psi, z, t, theta, pbuf, mubuf)) isa Float64
        @test (@inferred modelhessvec!(zeros(n), modelf, v)) isa Vector{Float64}
        @test (@inferred modelhessvec!(zeros(n), modelc, v)) isa Vector{Float64}
        @test (@inferred prox!(zeros(n), psi, z, t)) isa ROLOpt.DualSPGResult{Float64}
        @test (@inferred pgstep!(zeros(n), zeros(n), psi, xk, dgn, t)) isa
              ROLOpt.DualSPGResult{Float64}
        # dcauchy! previously inferred Tuple{Float64,Any,Any,Any,Float64}.
        @test (@inferred dcauchy!(zeros(n), modelf, psi, X, xk, dg, 1.0, 1.0, s0, n0,
                                  zeros(n), zeros(n), zeros(n), cnt, cp)) isa NTuple{5,Float64}
        @test (@inferred dspg!(copy(xk), modelf, psi, X, xk, dg, 1.0, s0, n0,
                               zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
                               cnt, sp)) isa NTuple{3,Float64}
    end

    @testset "@allocated == 0: hot kernels (type-stable barrier, post-warmup)" begin
        # warm up each call (compile the kernel + the nbytes specialization)
        nbytes(ROLOpt._dual_d!, psi, z, t, theta, pbuf, mubuf)
        nbytes(ROLOpt._dual_d!, psi_im, z, t, theta, pbuf, mubuf)
        nbytes(modelhessvec!, outn, modelf, v)
        nbytes(modelhessvec!, outn, modelc, v)
        nbytes(prox!, outn, psi, z, t)
        nbytes(pgstep!, outn, stepn, psi, xk, dgn, t)
        nbytes(dcauchy!, s, modelf, psi, X, xk, dg, 1.0, 1.0, s0, n0, px, dwa, dwa1, cnt, cp)
        nbytes(dspg!, copy(xk), modelf, psi, X, xk, dg, 1.0, s0, n0, gmodc, gmodp, stepn, pgi, hs, dx, cnt, sp)

        @test nbytes(ROLOpt._dual_d!, psi, z, t, theta, pbuf, mubuf) == 0
        @test nbytes(ROLOpt._dual_d!, psi_im, z, t, theta, pbuf, mubuf) == 0   # InexactMap(QuadraticMap) adjoint scratch
        @test nbytes(modelhessvec!, outn, modelf, v) == 0
        @test nbytes(modelhessvec!, outn, modelc, v) == 0          # F'' adjoint path
        @test nbytes(prox!, outn, psi, z, t) == 0
        @test nbytes(pgstep!, outn, stepn, psi, xk, dgn, t) == 0
        @test nbytes(dcauchy!, s, modelf, psi, X, xk, dg, 1.0, 1.0, s0, n0, px, dwa, dwa1, cnt, cp) == 0
        @test nbytes(dspg!, copy(xk), modelf, psi, X, xk, dg, 1.0, s0, n0, gmodc, gmodp, stepn, pgi, hs, dx, cnt, sp) == 0
    end

    @testset "@inferred: solve kernels (function barrier)" begin
        # solve() builds psi/model/(oracle,imap) behind a branch and dispatches once
        # to a typed kernel; each kernel must infer (its loop never dynamically
        # dispatches on a union-typed model/psi). Public solve() is allowed to branch.
        nk = 3
        fk = TikhonovObjective(1.0)
        Fk = AffineMap(randn(rng, 1, nk), [0.3])
        prob = RiskProblem(fk, Fk, ZeroProx{Float64}(), IntervalEnvelope(0.0, 1.0))

        optse = TROptions{Float64}(maxit=50)
        psie = PsiModel(prob.phi, prob.A, prob.F, prob.X, prob.Y, nk; opts=optse.dualspg)
        mdle = QuadraticModel(fk, prob.X, nk)
        rese = @inferred ROLOpt._solve_exact!(prob, optse, psie, mdle,
                                              ROLOpt._SolveBuffers(Float64, zeros(nk), 1), Counters())
        @test rese isa TRResult{Float64,Vector{Float64}}

        optsi = TROptions{Float64}(maxit=50, delta0=1.0)
        imap = InexactMap(prob.F, nk)
        psii = PsiModel(prob.phi, prob.A, imap, prob.X, prob.Y, nk; opts=optsi.dualspg)
        mdli = QuadraticModel(fk, prob.X, nk)
        resi = @inferred ROLOpt._solve_inexact!(prob, optsi, psii, mdli, InexactOracle(fk), imap,
                                                ROLOpt._SolveBuffers(Float64, zeros(nk), 1), Counters())
        @test resi isa TRResult{Float64,Vector{Float64}}
    end
end
