# Phase 1: the fraction-of-Cauchy-decrease condition (paper eq. for `pred`),
# evaluated with the SAME t0-based h_k used by the algorithm:
#   pred  >=  kappa_fcd * h_k * min( h_k / (1 + ||B||), Delta )   for some kappa_fcd > 0.
# Deterministic diagonal-quadratic + L1 instances; we assert a conservative
# kappa_fcd = 0.1 (observed ratios are >= ~0.32).

@testset "fraction-of-Cauchy-decrease bound" begin
    n = 4
    a = [1.0, 2.0, 3.0, 4.0]; b = [-1.0, 0.5, -2.0, 1.0]; wt = fill(0.3, n)
    f = DiagonalQuadratic(a, b)
    F = AffineMap(zeros(1, n), [0.0])
    phi = L1Objective(wt)
    A = IntervalEnvelope(0.0, 0.0)                  # risk inert
    X = EuclideanSpace(Float64); Y = EuclideanSpace(Float64)
    cp = CauchyParams{Float64}(1e-2, 0.1, 10.0, 10, 10, 1e-8)
    normB = maximum(a)                              # ||B|| for the diagonal model Hessian
    t0 = 1.0

    for (x0, del) in (([0.5, -0.3, 0.2, 0.1], 1.0),
                      ([0.0, 0.0, 0.0, 0.0], 0.5),
                      ([0.1, 0.1, 0.1, 0.1], 1e-2))
        x = collect(x0)
        psi = PsiModel(phi, A, F, X, Y, n); reanchor!(psi, x)
        model = QuadraticModel(f, X, n); cnt = Counters()
        gc = gradient(f, x); dg = similar(x); riesz!(dg, X, gc)
        setmodel!(model, x, gc)
        sval = value(f, x); nval = value(psi, x)
        step = zeros(n); pg = zeros(n)
        hk, hok = hstat!(step, pg, psi, X, x, dg, t0, cnt)
        s = zeros(n); px = zeros(n); dwa = zeros(n); dwa1 = zeros(n)
        _, svalc, nvalc, _, worstpg = dcauchy!(s, model, psi, X, x, dg, del, t0,
                                               sval, nval, px, dwa, dwa1, cnt, cp)
        pred = (sval + nval) - (svalc + nvalc)
        bound = hk * min(hk / (1 + normB), del)
        @test hok && worstpg < 1e-6     # h_k prox and all Cauchy proxes are accurate
        @test pred > 0
        @test pred >= 0.1 * bound
    end
end
