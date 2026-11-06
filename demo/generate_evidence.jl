# Proof-of-concept evidence generator for ROLOpt.jl.
#
# Runs the solver on problem instances *copied from the passing test suite* and
# emits plain data files (demo/data/*.dat) and LaTeX tabular fragments
# (demo/tables/*.tex) for the Beamer + pgfplots deck (demo/slides.tex). No
# plotting dependency: Julia produces numbers, LaTeX draws them.
#
# Run from anywhere:  julia --project=.. demo/generate_evidence.jl
# (uses only ROLOpt + LinearAlgebra + Random + Printf — all available).

using ROLOpt
using LinearAlgebra
using Random
using Printf

const HERE    = @__DIR__
const DATADIR = joinpath(HERE, "data")
const TABDIR  = joinpath(HERE, "tables")
mkpath(DATADIR); mkpath(TABDIR)

datapath(name) = joinpath(DATADIR, name)
tabpath(name)  = joinpath(TABDIR, name)

# Independent continuous-knapsack / water-filling reference for the E+CVaR support
# function (copied verbatim from test/test_riskenvelope.jl):
#   sup{ Σ wᵢθᵢyᵢ : Σ pᵢθᵢ = 1, l ≤ θᵢ ≤ u },  density = wᵢyᵢ/pᵢ.
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

# ---------------------------------------------------------------------------
# Slide 2 — finite-difference Taylor test of the package's nonlinear operator F
# (QuadraticMap) + the machine-precision identity checks the package ships.
# ---------------------------------------------------------------------------
function section_taylor()
    rng = MersenneTwister(2026)
    n = 4; m = 3
    Q = [(let M = randn(rng, n, n); (M'M) ./ n end) for _ in 1:m]   # symmetric PSD
    F = QuadraticMap(Q, randn(rng, m, n), randn(rng, m))
    x = randn(rng, n); v = randn(rng, n)

    Fx = mapvalue(F, x)                      # F(x)
    Jv = similar(Fx); applyJacobian!(Jv, F, v, x)   # F'(x) v
    Fxh = similar(Fx)

    # FORWARD differences exhibit the honest O(h) / O(h^2) Taylor slopes (central
    # differences would cancel exactly for a quadratic map -> the identity table).
    hs = exp10.(range(0.0, -8.0; length=33))
    open(datapath("jac_taylor.dat"), "w") do io
        println(io, "# h R0 R1")
        for h in hs
            @. Fxh = 0.0
            value!(Fxh, F, x .+ h .* v)
            R0 = norm(Fxh .- Fx)                 # ||F(x+hv)-F(x)||           ~ O(h)
            R1 = norm(Fxh .- Fx .- h .* Jv)      # ||... - h F'(x)v||         ~ O(h^2)
            @printf(io, "%.8e %.8e %.8e\n", h, max(R0, 1e-300), max(R1, 1e-300))
        end
    end

    # Package's own central-difference / identity checks (exact on quadratics).
    fobj = QuadraticObjective((let M = randn(rng, n, n); M'M end) + I, randn(rng, n))
    u = randn(rng, n); theta = randn(rng, m)
    eg = check_gradient(fobj, x, v)
    eh = check_hessvec(fobj, x, v)
    ej = check_jacobian(F, x, v)
    ea = check_adjoint(F, x, v, theta)
    es = check_hess_sym(fobj, x, u, v)
    open(tabpath("derivchecks.tex"), "w") do io
        println(io, "\\begin{tabular}{@{}lll@{}}")
        println(io, "\\toprule")
        println(io, "package check & identity & error\\\\")
        println(io, "\\midrule")
        @printf(io, "\\texttt{check\\_gradient}  & \$\\langle\\nabla f,v\\rangle=\$FD & \\num{%.2e}\\\\\n", eg)
        @printf(io, "\\texttt{check\\_hessvec}   & \$\\nabla^2\\! f\\,v=\$FD          & \\num{%.2e}\\\\\n", eh)
        @printf(io, "\\texttt{check\\_jacobian}  & \$F'(x)v=\$FD                     & \\num{%.2e}\\\\\n", ej)
        @printf(io, "\\texttt{check\\_adjoint}   & \$\\langle F'v,\\theta\\rangle=\\langle v,F'^{*}\\theta\\rangle\$ & \\num{%.2e}\\\\\n", ea)
        @printf(io, "\\texttt{check\\_hess\\_sym} & \$\\langle Hu,v\\rangle=\\langle Hv,u\\rangle\$ & \\num{%.2e}\\\\\n", es)
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    return (; eg, eh, ej, ea, es)
end

# ---------------------------------------------------------------------------
# Slide 3 — closed-form optima (LASSO across 3 subsolvers, README -1/6) and the
# precision ladder (Float32 / Float64 / BigFloat under precision-scaled tols).
# ---------------------------------------------------------------------------
# ROL TypeP LASSO test_01.cpp reduction (test/test_lasso_reduce.jl, seed 123).
function lasso_instance()
    rng = MersenneTwister(123); n = 5
    a = 5 .* rand(rng, n) .+ 1e-2
    b = 20 .* rand(rng, n) .- 10
    w = rand(rng, n)
    y = 10 .* rand(rng, n) .- 5
    prob = RiskProblem(DiagonalQuadratic(a, b), AffineMap(zeros(1, n), [0.0]),
                       L1Objective(w, y), IntervalEnvelope(0.0, 0.0))
    xstar = @. (clamp(a * y + b, -w, w) - b) / a
    return prob, xstar, n
end

# min ½‖x‖² + max(0, Σx + 0.5), x* = -1/6·ones (test/test_precision.jl).
risk1(::Type{T}, n) where {T} = RiskProblem(TikhonovObjective(one(T)),
    AffineMap(ones(T, 1, n), T[0.5]), ZeroProx{T}(), IntervalEnvelope(zero(T), one(T)))

# The LASSO instance over a working precision T (Float64 draws, converted to T),
# with the ROL closed form at T. Used for the precision ladder.
function lasso_T(::Type{T}) where {T}
    rng = MersenneTwister(123); n = 5
    a = T.(5 .* rand(rng, n) .+ 1e-2); b = T.(20 .* rand(rng, n) .- 10)
    w = T.(rand(rng, n)); y = T.(10 .* rand(rng, n) .- 5)
    prob = RiskProblem(DiagonalQuadratic(a, b), AffineMap(zeros(T, 1, n), T[0.0]),
                       L1Objective(w, y), IntervalEnvelope(zero(T), zero(T)))
    xstar = @. (clamp(a * y + b, -w, w) - b) / a
    return prob, xstar, n
end

# Format an error for a table: collapse exact-zero (machine-exact recovery) to a
# "< machine eps" label rather than a misleading "0.00e+00".
fmterr(err, T=Float64) = err > 0 ? (@sprintf("\\num{%.2e}", err)) : (@sprintf("\$<\\num{%.0e}\$", eps(T)))

function section_closedform()
    rows = Tuple{String,String,Int,String,Float64}[]   # problem, solver, iters, err-str, hk

    prob, xstar, n = lasso_instance()
    for s in (:cauchy, :spg, :ncg)
        res = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=5000, subsolver=s))
        push!(rows, ("LASSO (ROL \\texttt{test\\_01})", String(s), res.iters,
                     fmterr(norm(res.x - xstar)), res.hk))
    end

    # README kink optimum at Float64 — recovered exactly (a second closed form).
    r64 = solve(risk1(Float64, 3), zeros(3); opts=TROptions{Float64}())
    push!(rows, ("kink \$x^\\star=-\\tfrac16\\mathbf 1\$", "cauchy", r64.iters,
                 fmterr(norm(r64.x .- (-1 / 6))), r64.hk))

    open(tabpath("closedform.tex"), "w") do io
        println(io, "\\begin{tabular}{@{}llrll@{}}")
        println(io, "\\toprule")
        println(io, "problem & subsolver & iters & \$\\lVert x-x^\\star\\rVert\$ & \$h_k\$\\\\")
        println(io, "\\midrule")
        for (p, s, it, errs, hk) in rows
            @printf(io, "%s & \\texttt{:%s} & %d & %s & \\num{%.2e}\\\\\n", p, s, it, errs, hk)
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    return rows
end

# Precision ladder: the SAME generic LASSO solved at Float32/Float64/BigFloat under
# precision-scaled tolerances. Achieved ‖x-x*‖ improves with the requested tol.
function section_ladder()
    out = Tuple{Int,String,Float64,Float64,Float64}[]
    specs = ((1, Float32, 1.0f-3, 1.0f-5),
             (2, Float64, 1e-8, 1e-12),
             (3, BigFloat, BigFloat(1e-20), BigFloat(1e-28)))
    open(datapath("precision_ladder.dat"), "w") do io
        println(io, "# idx err htol macheps")
        for (idx, T, htol, dtol) in specs
            prob, xstar, n = lasso_T(T)
            res = solve(prob, zeros(T, n);
                        opts=TROptions{T}(maxit=5000, subsolver=:spg, htol=T(htol),
                              dualspg=DualSPGOptions{T}(tol=T(dtol))))
            err = Float64(norm(res.x - xstar))
            @printf(io, "%d %.6e %.6e %.6e\n", idx, err, Float64(htol), Float64(eps(T)))
            push!(out, (idx, string(T), err, Float64(htol), res.converged ? 1.0 : 0.0))
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Slide 4 — finite-dimensional E+CVaR analogue: support function vs the
# independent water-filling reference, and h_k convergence per subsolver.
# ---------------------------------------------------------------------------
function section_cvar()
    # suppA vs water-filling reference (test/test_riskenvelope.jl, seed 2).
    rng = MersenneTwister(2); n = 8
    probs = (p = rand(rng, n) .+ 0.1; p ./ sum(p))
    lambda, pq = 0.75, 0.9
    A = ConvexCombinationEnvelope(lambda, pq, probs)
    Yprob = WeightedSpace(probs); Xeuc = EuclideanSpace(Float64)
    maxabs_p = 0.0; maxabs_e = 0.0
    for _ in 1:30
        y = randn(rng, n)
        maxabs_p = max(maxabs_p, abs(suppA(A, Yprob, y) - greedy_suppA(lambda, pq, probs, y, probs)))
        maxabs_e = max(maxabs_e, abs(suppA(A, Xeuc, y) - greedy_suppA(lambda, pq, probs, y, ones(n))))
    end
    open(tabpath("cvar_suppA.tex"), "w") do io
        println(io, "\\begin{tabular}{@{}ll@{}}")
        println(io, "\\toprule")
        println(io, "\$Y\$ metric & \$\\max_y|\\sigma_A-\\sigma_A^{\\mathrm{ref}}|\$\\\\")
        println(io, "\\midrule")
        @printf(io, "probability \$L^2(P)\$ & \\num{%.2e}\\\\\n", maxabs_p)
        @printf(io, "Euclidean            & \\num{%.2e}\\\\\n", maxabs_e)
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end

    # h_k convergence: shared CVaR instance (seed 5), re-solved at maxit=1..K so the
    # exact path yields a true per-iteration trajectory (deterministic).
    srng = MersenneTwister(5); nn = 3; N = 4
    cprobs = (p = rand(srng, N) .+ 0.2; p ./ sum(p))
    Yp = WeightedSpace(cprobs)
    Q = [let Mt = randn(srng, nn, nn); (Mt'Mt) ./ nn .* 0.05 end for _ in 1:N]
    F = QuadraticMap(Q, randn(srng, N, nn), randn(srng, N))
    Acv = ConvexCombinationEnvelope(0.75, 0.9, cprobs)
    cprob = RiskProblem(TikhonovObjective(0.2), F, ZeroProx{Float64}(), Acv; Y=Yp)
    K = 60
    traj = Dict{Symbol,Vector{Float64}}()
    for s in (:cauchy, :spg, :ncg)
        traj[s] = [solve(cprob, zeros(nn); opts=TROptions{Float64}(maxit=k, subsolver=s)).hk for k in 1:K]
    end
    open(datapath("hk_convergence.dat"), "w") do io
        println(io, "# iter hk_cauchy hk_spg hk_ncg")
        for k in 1:K
            @printf(io, "%d %.6e %.6e %.6e\n", k, traj[:cauchy][k], traj[:spg][k], traj[:ncg][k])
        end
    end
    return (; maxabs_p, maxabs_e, hk_cauchy_end=traj[:cauchy][end],
              hk_ncg_end=traj[:ncg][end])
end

# ---------------------------------------------------------------------------
# Slide 5 — inexact evaluations (Conditions 1 & 2) audited per iteration, and
# convergence to the exact minimizer (test/test_inexact.jl, seed 202, kappa=1).
# ---------------------------------------------------------------------------
function section_inexact()
    srng = MersenneTwister(202); n = 5
    f = TikhonovObjective(0.5)
    F = AffineMap(randn(srng, 1, n), [0.3])
    phi = L1Objective(fill(1e-2, n))
    prob = RiskProblem(f, F, phi, IntervalEnvelope(0.0, 1.0))
    oracle = InexactOracle(f; kgrad=1.0, kval=1.0, kjac=1.0, kobj=1e2, rho=0.5)
    res = solve(prob, zeros(n);
                opts=TROptions{Float64}(maxit=2000, subsolver=:spg, delta0=10.0), oracle=oracle)
    re = solve(prob, zeros(n); opts=TROptions{Float64}(maxit=2000, subsolver=:spg))

    maxratio = 0.0   # worst realized-error / bound across all iterations & conditions
    open(datapath("inexact_records.dat"), "w") do io
        println(io, "# iter hk tolg eg tolb eb tolj eA credbound absdiff")
        for r in oracle.records
            ad = abs(r.ared - r.cred)
            for (e, t) in ((r.eg, r.tolg), (r.eb, r.tolb), (r.eA, r.tolj), (ad, r.credbound))
                t > 0 && (maxratio = max(maxratio, e / t))
            end
            @printf(io, "%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                    r.iter, r.hk, r.tolg, r.eg, r.tolb, r.eb, r.tolj, r.eA, r.credbound, ad)
        end
    end

    matcherr = norm(res.x - re.x)
    open(tabpath("inexact.tex"), "w") do io
        println(io, "\\begin{tabular}{@{}ll@{}}")
        println(io, "\\toprule")
        println(io, "quantity & value\\\\")
        println(io, "\\midrule")
        @printf(io, "outer iterations (audited) & %d\\\\\n", length(oracle.records))
        @printf(io, "worst realized/bound ratio & %.2f \$\\le 1\$\\\\\n", maxratio)
        @printf(io, "\$\\lVert x_{\\text{inexact}}-x_{\\text{exact}}\\rVert\$ & \\num{%.2e}\\\\\n", matcherr)
        @printf(io, "final \$h_k\$ & \\num{%.2e}\\\\\n", res.hk)
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    return (; nrec=length(oracle.records), maxratio, matcherr, hk=res.hk)
end

function main()
    println("ROLOpt.jl evidence generator")
    println("  data -> $DATADIR")
    println("  tables -> $TABDIR\n")

    t = section_taylor()
    @printf("[slide 2] Taylor/identity checks: grad %.1e hess %.1e jac %.1e adj %.1e sym %.1e\n",
            t.eg, t.eh, t.ej, t.ea, t.es)

    rows = section_closedform()
    println("[slide 3] closed-form rows:")
    for r in rows
        @printf("    %-32s :%-7s iters=%4d  hk=%.2e\n", r[1], r[2], r[3], r[5])
    end
    ladder = section_ladder()
    println("[slide 3] precision ladder (LASSO, :spg):")
    for r in ladder
        @printf("    %-9s htol=%.0e  err=%.3e  converged=%s\n", r[2], r[4], r[3], r[5] == 1.0)
    end

    c = section_cvar()
    @printf("[slide 4] suppA err: prob %.2e euclid %.2e | hk_cauchy(end)=%.2e hk_ncg(end)=%.2e\n",
            c.maxabs_p, c.maxabs_e, c.hk_cauchy_end, c.hk_ncg_end)

    ix = section_inexact()
    @printf("[slide 5] inexact: %d records, worst ratio %.3f, match err %.2e, final hk %.2e\n",
            ix.nrec, ix.maxratio, ix.matcherr, ix.hk)

    println("\nAll data/tables written.")
end

main()
