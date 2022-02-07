module EigenvalueSolver

# A Julia package for solving systems of polynomial equations using eigenvalue methods.
# Matías R. Bender and Simon Telen - 14-05-2021

using DynamicPolynomials
using LinearAlgebra
using GenericSchur
using GenericSVD
using Statistics
using SmithNormalForm  # to add: pkg> add https://github.com/wildart/SmithNormalForm.jl.git

import TaylorSeries


include("PolytopeUtils.jl")

# Main function.
# Solves the system f = 0 in the variables x using the Sylvester map defined by A₀, E, D.
# INPUT:
# f     vector of polynomials of length s
# x     array of variables
# A₀    array representing the exponents of the monomials used to construct generalized multiplication matrices
# E     vector of arrays of length s+1, represents the monomial multipliers of f₀, f₁, …, fs (rows are exponent vectors)
# D     array representing a monomial basis of the codomain of the Sylvester map (rows are exponent vectors)
# OUTPUT:
# a set of candidate solutions to f(x) = 0.
# ------------------ It is assumed that (f,A₀,E,D) is an admissible tuple
# ------------------ and that all arrays in A₀, E, D have the same number of columns (n)
# ------------------ If the cokernel N of the Sylvester map was already computed, it is possible to pass N as an input
function solve_EV(
    f,
    x,
    A₀,
    E,
    D;
    rankTol = 1e3 * eps(), #tolerance for rank determination and nullspace computation
    complex = true,
    check_criterion = true,
    EVtol = 1e-4, #tolerance for extracting eigenvalues
    N = nothing,
    compress = true, #compress Sylv before computing cokernel
    schurfac = false, #compute eigenvalues via Schur factorization
    filter = false, #filter the solutions at the end
    filtertol = 1e-6,
    verbose::Bool=false, #print info
)
    x = variables(f)
    n = length(x) # number of variables
    s = length(f) # number of equations, s ≥ n
    #-------------------------------------------------------------------------
    Σ = exptomon(D, x)
    if N == nothing
        verbose && println("constructing Sylvester matrix...")
        σ = [exptomon(e, x) for e ∈ E[2:end]]
        @time Sylv = getRes(f, Σ, σ, x; complex = complex)
        verbose && println("       Sylv is a matrix of size $(size(Sylv))")
        #-------------------------------------------------------------------------
        verbose && println("computing cokernel...")
        if compress && size(Sylv, 2) > size(Sylv, 1)
            verbose && println("compressing Sylv...")
            @time Sylv = Sylv * randn(size(Sylv, 2), size(Sylv, 1))
        end
        @time svdobj = svd(transpose(Sylv), full = true)
        firstzerosingval = findfirst(sv -> sv < rankTol * svdobj.S[1], svdobj.S)
        if !isnothing(firstzerosingval)
            rk = firstzerosingval - 1
            verbose && println("       rank = $(rk)")
            verbose && println("       relative size of last nz singular value = $(svdobj.S[rk]/svdobj.S[1])")
            verbose && println("       gap = $(svdobj.S[rk]/svdobj.S[rk+1])")
        else
            rk = length(svdobj.S)
            verbose && println("       relative size of last nz singular value = $(svdobj.S[rk]/svdobj.S[1])")
        end
        N = transpose(svdobj.V[:, rk+1:end])
    end
    verbose && println("N has size $(size(N))")
    #-------------------------------------------------------------------------
    Σ₀ = exptomon(E[1], x)
    Σα0 = exptomon(A₀, x)
    if check_criterion
        verbose && println("Checking criterion...")
        chck = checkCriterion(Σα0, Σ₀, Σ, N, x; verbose = verbose)
    end
    #-------------------------------------------------------------------------
    if !check_criterion || chck
        verbose && println("finding a basis using QR-P...")
        @time M, κ = get_multmtces_QR_toric(N, Σ, Σ₀, Σα0, x; complex = complex)
        #-------------------------------------------------------------------------
        verbose && println("extracting relevant eigenvalues...")
        if schurfac
            @time monsol =
                simulDiag_schur_toric(M; clustering = true, tol = 1e-4, complex = complex)
        else
            @time monsol = extractSolutions_fast(M, Σα0; tol = EVtol)
        end

        #-------------------------------------------------------------------------
        verbose && println("inverting monomial map given by A₀...")
        A₀2 = convert(Array{Int64,2}, A₀')
        zeroind = findfirst(ℓ -> norm(A₀2[:, ℓ]) == 0, 1:size(A₀2, 2))
        @time logsol, sol = solvebinom_SNF_array(A₀2, [ms / ms[zeroind] for ms ∈ monsol])
        verbose && println("found $(length(sol)) solutions")
        if filter
            verbose && println("filtering...")
            res = get_residual(f, sol, x)
            sol = sol[findall(r -> r < filtertol, res)]
        end
        return sol
    else
        println("Criterion was not satisfied.")
    end
end

# Returns a vector of monomials in x corresponding to the exponent tuples in the rows of M.
function exptomon(M, x)
    mons = fill(prod(x), size(M, 1))
    mtx = x' .^ M
    mons = map(k -> prod(mtx[k, :]), 1:size(mtx, 1)) # this gives a vector of monomials
end

# Converts an array of monomials into an array of exponents.
function montoexp(Σ)
    exparray = exponents.(Σ)
    M = fill(0, length(Σ), length(exparray[1]))
    for i = 1:length(Σ)
        M[i, :] = exparray[i]
    end
    return M
end

# Computes the Sylvester map in the monomial bases σ[i] for R_{A_i} and Σ for R_D.
# x should contain the variables of the system.
function getRes(f, Σ, σ, x; complex = false)
    if complex
        res = fill(0.0 + 0.0 * im, length(Σ), sum(map(i -> length(i), σ)))
    else
        res = fill(0.0, length(Σ), sum(map(i -> length(i), σ)))
    end
    n = length(x)
    s = length(f)
    mapping = Dict(Σ .=> 1:length(Σ))
    for i = 1:s
        J = sum(map(k -> length(k), σ[1:i-1]))
        for j = 1:length(σ[i])
            pol = σ[i][j] * f[i]
            res[map(k -> mapping[k], monomials(pol)), J+j] = coefficients(pol)
        end
    end
    return res
end

# Check if N_{f₀} is of rank rank(N) for generic elements f₀ in span(Σα0).
# Here N is the previously computed cokernel of a Sylvester map with support D.
# The monomials in Σ₀ should correspond to E₀ and those in Σ to D.
function checkCriterion(Σα0, Σ₀, Σ, N, x; rankTol = 1e3 * eps(), verbose::Bool=false)
    f₀ = randn(ComplexF64, length(Σα0))' * Σα0
    M0 = getRes([f₀], Σ, [Σ₀], x; complex = true)
    r = rank(N * M0; rtol = rankTol)
    if r == size(N, 1)
        verbose && println("********* criterion satisfied ********* γ = ", r)
        return true
    else
        verbose && println("!!!!!!!!! criterion violated !!!!!!!!!")
        return false
    end
end

# This computes the multiplication matrices via a basis selection which uses QR with optimal column pivoting
function get_multmtces_QR_toric(N, Σ_α_α₀, Σ_α, Σ_α₀, t; complex = false)
    δ = size(N, 1)
    mapping = Dict(Σ_α_α₀ .=> 1:length(Σ_α_α₀))
    n = length(t)
    n_α₀ = length(Σ_α₀)
    if complex
        Nᵢ = fill(fill(0.0 + 0.0 * im, δ, length(Σ_α)), n_α₀)
        M = fill(fill(0.0 + 0.0 * im, δ, δ), n_α₀) # This will contain the multiplication operators corresponding to the variables
    else
        Nᵢ = fill(fill(0.0, δ, length(Σ_α)), n_α₀)
        M = fill(fill(0.0, δ, δ), n_α₀) # This will contain the multiplication operators corresponding to the variables
    end
    for i = 1:n_α₀
        indsᵢ = map(k -> mapping[k*Σ_α₀[i]], Σ_α)
        Nᵢ[i] = N[:, indsᵢ]
    end
    if complex
        randcoeff = randn(ComplexF64, n_α₀)
        Nh₀ = sum( randcoeff.* Nᵢ)
    else
        Nh₀ = sum(randn(n_α₀) .* Nᵢ) # Dehomogenization with respect to a generic linear form.
    end
    QRobj = qr(Nh₀, Val(true))
    pivots = QRobj.p
    Nb = triu(QRobj.R[1:δ, 1:δ])
    κ = cond(Nb)
    if κ > 1e12
        println("warning: this might not be zero-dimensional in X")
    end
    for i = 1:n_α₀
        M[i] = Nb \ (QRobj.Q' * Nᵢ[i][:, pivots[1:δ]])
    end
    return M, κ
end

# This computes an upper triangularized, column pivoted version of N via a pivoted QR factorization of
# the map N: R_D → C^γ whose columns are indexed by the monomials in Σ_α_α₀ ~ D.
# the monomials in the subspace W (largest subspace s.t. W⁺ ⊂ V) are in Σ_α ~ E₀.
function get_QRbasis_general(N, Σ_α_α₀, Σ_α)
    mapping = Dict(Σ_α_α₀ .=> 1:length(Σ_α_α₀))
    δ = size(N, 1)
    indsW = map(k -> mapping[k], Σ_α)
    indsNotW = findall(i -> i ∉ indsW, 1:length(Σ_α_α₀))
    QRobj = qr(N[:, indsW], Val(true))
    pivots = QRobj.p
    N = triu(QRobj.Q' * N[:, vcat(indsW[pivots], indsNotW)])
    Σ = Σ_α_α₀[vcat(indsW[pivots], indsNotW)]
    return N, Σ
end

# simultaneous diagonalization of the matrices in M via Schur factorization of a random linear combination.
# returns an array with the solutions (`third mode` of the CPD)
function simulDiag_schur_toric(M; clustering = false, tol = 1e-4, complex = false)
    solMatrix =
        simulDiag_schur_mtx(M; clustering = clustering, tol = tol, complex = complex)
    δ = size(solMatrix, 1)
    monsol = map(i -> solMatrix[i, :], 1:δ)
    return monsol
end

# This is a function for clustering the eigenvalues.


# Match the points in sol to the points in refsol by permuting them as defined by 'usedinds'
# and compute the relative distance between the matched points.
function getForwardError(sol, refsol)
    errs = fill(0.0, 0)
    usedinds = fill(1, 0)
    δ = length(refsol)
    for s ∈ sol
        remaininginds = setdiff(1:δ, usedinds)
        if length(remaininginds) > 1
            nrms = norm.([(ref - s)/norm(ref) for ref ∈ refsol[remaininginds]])
            amin = argmin(nrms)
            matchind = remaininginds[amin]
            push!(usedinds, matchind)
            push!(errs, nrms[amin])
        else
            push!(usedinds, remaininginds[1])
            push!(errs, norm(refsol[remaininginds[1]] - s)/norm(refsol[remaininginds[1]]))
        end
    end
    return errs, usedinds
end

# Extracts the useful eigenvalues from an array M of matrices M_{x^α} for all
# monomials in x^α ∈ Σα0.
function extractSolutions_fast(M, Σα0; tol = 1e-5, clustertol = 1e-7)
    Mg = sum([randn(ComplexF64) * MM for MM ∈ M])
    Mgprime = sum([randn(ComplexF64) * MM for MM ∈ M])
    eigenobj = eigen(transpose(Mg))
    eigenvals = eigenobj.values
    eigenvecs = eigenobj.vectors
    clusters, inds = getClusters(eigenvals; tol = clustertol)
    monsol = []
    γ = size(Mg, 1)
    k = 1
    for i = 1:length(clusters)
        λ = mean(clusters[i])
        Vλ = eigenvecs[:, inds[i]]
        if size(Vλ, 2) > 0
            if size(Vλ, 2) == 1
                aux = [transpose(Vλ) * Mgprime; transpose(Vλ)]
                svdobj = svd(aux)
                ratio = svdobj.S[2] / svdobj.S[1]
                if ratio < tol
                    eval = [mean(aux[1, :] ./ aux[2, :])]
                    espace = [[1]]
                else
                    eval = []
                    espace = []
                end
            else
                Vλ, = svd(Vλ)
                eval, espace = solveREP(transpose(Vλ) * Mgprime, transpose(Vλ); rtol = tol)
            end
        end
        if length(eval) >= 1
            Vλnew = vcat(espace...)
            Vλ = Vλ * transpose(Vλnew)
            m = size(Vλ, 2)
            if m == 1 && !isnan(eval[1])
                monsol = push!(monsol, [(transpose(Vλ)*MM*Vλ)[1] for MM ∈ M])
            elseif m > 1 && !isnan(eval[1])
                O = randn(ComplexF64, γ, m)
                aux = fill(0.0 + 0.0 * im, length(M))
                for i = 1:length(M)
                    aux[i] = tr((transpose(Vλ) * M[i] * O / (transpose(Vλ) * O))) / m
                end
                monsol = push!(monsol, aux)
            end
        end
        k += 1
    end
    return monsol
end

# notation of `Numerical Root Finding via Cox Rings`
# solves the binomial systems {xᵃ = Λ[i]ⱼ} for i = 1...length(Λ) and a is the j-th column of A.
function solvebinom_SNF_array(A, Λ)
    F = smith(A)
    k = size(A, 1)
    δ = length(Λ)
    n_α₀ = size(A, 2)
    invfactors = F.SNF
    r = findfirst(k -> k == 0, invfactors)  #rank
    if isnothing(r)
        r = minimum(size(A))
    else
        r = r - 1
    end
    U = inv(F.S)
    V = inv(F.T)
    Λmat = fill(0.0 + 0.0 * im, length(Λ), n_α₀)
    for i = 1:δ
        Λmat[i, :] = Λ[i]
    end
    w = log.(Complex.(Λmat)) * V[:, 1:r]
    w = w * diagm(1 ./ invfactors[1:r])
    logY = hcat(w, zeros(δ, k - r))
    logX = logY * U
    loghomsol = map(k -> logX[k, :], 1:δ)
    homsol = map(k -> exp.(loghomsol[k]), 1:δ)
    return loghomsol, homsol
end

# Compute the cokernel from a larger Sylvester map, supported on (oldsupport ∪ newsupport), from
# the cokernel 'oldcokernel' of a Sylvester map supported on oldsupport.
# The monomial multipliers of the equations in f to be added are given by an array `newshifts' of monomial vectors.
function updateCokernel(
    f,
    x,
    oldcokernel,
    oldsupport,
    newsupport,
    newshifts;
    complex = false,
    rankTol = 1e3 * eps(),
)
    newsylvprime = getRes(f, newsupport, newshifts, x; complex = complex)
    oldδ = size(oldcokernel, 1)
    growth = length(newsupport) - length(oldsupport)
    mapping = Dict(newsupport .=> 1:length(newsupport))
    oldinds = [mapping[ind] for ind ∈ oldsupport]
    newinds = setdiff(1:length(newsupport), oldinds)
    id = Matrix(I, growth, growth)
    if complex
        prodmtx = fill(0.0 + 0.0im, oldδ + growth, length(newsupport))
    else
        prodmtx = fill(0.0, oldδ + growth, length(newsupport))
    end
    prodmtx[1:oldδ, oldinds] = oldcokernel
    prodmtx[oldδ+1:end, newinds] = id
    L = transpose(nullspace(transpose(prodmtx * newsylvprime), rtol = rankTol))
    return L * prodmtx
end

# Compute an admissible tuple for a dense system of polynomials f by incrementing the degree
# until our criterion is satisfied.
function findAdmissibleTuple_dense(f, x, DMAX; complex = false, rankTol = 1e3 * eps(), verbose::Bool=false, trySemiregular::Bool=false)
    s = length(f)
    ds = fill(0, s)
    for i = 1:s
        ds[i] = maximum(degree.(terms(f[i])))
    end
    if trySemiregular
        varTaylor = TaylorSeries.Taylor1(Integer,sum(ds))
        HScandidate = prod(vv -> (1-varTaylor^vv),ds)/(1-varTaylor)^length(x)
        dmax = max(findfirst(vv -> vv <= 0,HScandidate.coeffs)-1,maximum(ds))
    else
        dmax = maximum(ds)
    end
    verbose && println("degree = ", dmax)
    oldsupport = getMonomials(x, dmax)
    oldshifts = [getMonomials(x, dmax - ds[i]) for i = 1:s]
    Σα0 = getMonomials(x, 1)
    Σ₀ = getMonomials(x, dmax - 1)
    oldSylv = getRes(f, oldsupport, oldshifts, x; complex = complex)
    oldcokernel = transpose(nullspace(transpose(oldSylv), rtol = rankTol))
    chck = checkCriterion(Σα0, Σ₀, oldsupport, oldcokernel, x; verbose = verbose)
    while dmax < DMAX && !chck
        dmax = dmax + 1
        verbose && println("degree = ", dmax)
        newsupport = getMonomials(x, dmax)
        newshifts = [getMonomials(x, dmax - ds[i]; homogeneous = true) for i = 1:s]
        newcokernel = updateCokernel(
            f,
            x,
            oldcokernel,
            oldsupport,
            newsupport,
            newshifts;
            complex = complex,
            rankTol = rankTol,
        )
        testshifts = [getMonomials(x, dmax - ds[i]; homogeneous = false) for i = 1:s]
        oldsupport = newsupport
        oldcokernel = newcokernel
        Σ₀ = getMonomials(x, dmax - 1)
        chck = checkCriterion(Σα0, Σ₀, oldsupport, oldcokernel, x; verbose = verbose)
    end
    topdeg = dmax
    A₀ = montoexp(getMonomials(x, 1))
    extdeg = topdeg .- ds
    extdeg = [topdeg - 1; extdeg]
    E = [montoexp(getMonomials(x, extdeg[i])) for i = 1:length(f)+1]
    D = montoexp(getMonomials(x, topdeg))
    return oldcokernel, oldsupport, chck, dmax, (A₀, E, D)
end

# Compute an admissible tuple for a sparse unmixed system of polynomials f by incrementing the degree
# until our criterion is satisfied.
function findAdmissibleTuple_unmixed(
    f,
    x,
    A,
    α,
    DMAX;
    complex = false,
    rankTol = 1e3 * eps(),
    verbose::Bool=false
)
    s = length(f)
    dmax = maximum(α)
    A₀pol = sum([prod(x .^ (A[i, :])) for i = 1:size(A, 1)])
    A₀polytope = newtonPolytope(A₀pol, x)
    vtcs = vertexRepresentation(A₀polytope)
    A₀pol = sum([prod(x .^ (vtcs[i, :])) for i = 1:size(vtcs, 1)])
    A₀polytope = newtonPolytope(A₀pol, x)
    Aold = newtonPolytope(A₀pol^dmax, x)
    oldlatticepoints = getLatticePoints(Aold)
    oldsupport = exptomon(oldlatticepoints, x)
    oldpolytopes = [newtonPolytope(A₀pol^(dmax - α[i]), x) for i = 1:s]
    oldshiftlatticepoints = [getLatticePoints(P) for P ∈ oldpolytopes]
    oldshifts = [exptomon(oslp, x) for oslp ∈ oldshiftlatticepoints]
    Σα0 = exptomon(getLatticePoints(A₀polytope), x)
    Σ₀ = exptomon(getLatticePoints(newtonPolytope(A₀pol^(dmax - 1), x)), x)
    oldSylv = getRes(f, oldsupport, oldshifts, x; complex = complex)
    oldcokernel = transpose(nullspace(transpose(oldSylv), rtol = rankTol))
    chck = checkCriterion(Σα0, Σ₀, oldsupport, oldcokernel, x; verbose = verbose)
    while dmax < DMAX && !chck
        dmax = dmax + 1
        verbose && println("degree = ", dmax)
        newsupport = exptomon(getLatticePoints(newtonPolytope(A₀pol^(dmax), x)), x)
        newshifts = [
            setdiff(
                exptomon(getLatticePoints(newtonPolytope(A₀pol^(dmax - α[i]), x)), x),
                exptomon(getLatticePoints(newtonPolytope(A₀pol^(dmax - α[i] - 1), x)), x),
            ) for i = 1:s
        ]
        newcokernel = updateCokernel(
            f,
            x,
            oldcokernel,
            oldsupport,
            newsupport,
            newshifts;
            complex = complex,
        )
        oldsupport = newsupport
        oldcokernel = newcokernel
        Σ₀ = exptomon(getLatticePoints(newtonPolytope(A₀pol^(dmax - 1), x)), x)
        chck = checkCriterion(Σα0, Σ₀, oldsupport, oldcokernel, x; rankTol = rankTol, verbose = verbose)
    end
    topdeg = dmax
    extdeg = topdeg .- α
    extdeg = [topdeg - 1; extdeg]
    Eholes = [extdeg[i] * A for i = 1:length(extdeg)]
    Dholes = topdeg * A
    Epols = [sum([prod(x .^ (AA[i, :])) for i = 1:size(AA, 1)]) for AA ∈ Eholes]
    Dpol = sum([prod(x .^ (Dholes[i, :])) for i = 1:size(Dholes, 1)])
    Epolytopes = [newtonPolytope(ff, x) for ff ∈ Epols]
    Dpolytope = newtonPolytope(Dpol, x)
    E = [getLatticePoints(P) for P ∈ Epolytopes]
    D = getLatticePoints(Dpolytope)
    A₀pol = sum([prod(x .^ (A[i, :])) for i = 1:size(A, 1)])
    A₀polytope = newtonPolytope(A₀pol, x)
    A₀ = getLatticePoints(A₀polytope)
    return oldcokernel, oldsupport, chck, dmax, (A₀, E, D)
end


# Compute the kernel of a dense Sylvester map  supported in degree DMAX by using the 'degree by degree' procedure.
function getKernel_dense_DBD(f, x, DMAX; complex = false, rankTol = 1e3 * eps(), verbose::Bool=false)
    verbose && println("Computing cokernel degree by degree...")
    s = length(f)
    ds = fill(0, s)
    for i = 1:s
        ds[i] = maximum(degree.(terms(f[i])))
    end
    dmax = maximum(ds)
    oldsupport = getMonomials(x, dmax)
    oldshifts = [getMonomials(x, dmax - ds[i]) for i = 1:s]
    Σα0 = getMonomials(x, 1)
    Σ₀ = getMonomials(x, dmax - 1)
    oldSylv = getRes(f, oldsupport, oldshifts, x; complex = complex)
    oldcokernel = transpose(nullspace(transpose(oldSylv), rtol = rankTol))
    while dmax < DMAX
        dmax = dmax + 1
        verbose && println("degree = ", dmax)
        newsupport = getMonomials(x, dmax)
        newshifts = [getMonomials(x, dmax - ds[i]; homogeneous = true) for i = 1:s]
        newcokernel = updateCokernel(
            f,
            x,
            oldcokernel,
            oldsupport,
            newsupport,
            newshifts;
            complex = complex,
        )
        testshifts = [getMonomials(x, dmax - ds[i]; homogeneous = false) for i = 1:s]
        oldsupport = newsupport
        oldcokernel = newcokernel
        Σ₀ = getMonomials(x, dmax - 1)
    end
    return oldcokernel, oldsupport, dmax
end

# Compute an admissible tuple for a (square,) dense system of equations.
# This is based on the Macaulay bound for the regularity.
function get_AT_CI_dense(f, x)
    s = length(f)
    n = length(x)
    ds = fill(0, s)
    for i = 1:s
        ds[i] = maximum(degree.(terms(f[i])))
    end
    dreg = sum(ds) - n + 1
    Dmons = getMonomials(x, dreg)
    Emons = [getMonomials(x, dreg - ds[i]) for i = 1:s]
    Emons = [[getMonomials(x, dreg - 1)]; Emons]
    A₀mons = getMonomials(x, 1)
    D = montoexp(Dmons)
    E = [montoexp(EE) for EE ∈ Emons]
    A₀ = montoexp(A₀mons)
    return A₀, E, D
end

# Compute an admissible tuple for a (square,) multi-graded dense system of equations.
# This is based on the Macaulay bound for the regularity.
function get_AT_CI_multi_dense(f, vargroups, ds)
    s = size(ds, 1)
    dreg = sum(ds, dims = 1) - length.(vargroups)' .+ 1
    Dmons = getMultiMonomials(vargroups, dreg)
    Emons = [getMultiMonomials(vargroups, dreg - ds[i, :]') for i = 1:s]
    Emons = [[getMultiMonomials(vargroups, dreg - ones(Int, length(dreg))')]; Emons]
    A₀mons = getMultiMonomials(vargroups, ones(Int, length(dreg))')
    D = montoexp(Dmons)
    E = [montoexp(EE) for EE ∈ Emons]
    A₀ = montoexp(A₀mons)
    return A₀, E, D
end

# Compute an admissible tuple for a (square,) multi-unmixed system of equations.
# This is based on the Macaulay bound for the regularity.
function get_AT_CI_multi_unmixed(f, vargroups, sups, ds)
    s = size(ds, 1)
    codegs = [getCodegree(sups[i]) for i = 1:length(vargroups)]
    dreg = sum(ds, dims = 1) - codegs' .+ 2
    Dmons = getMultiUnmixedMonomials(vargroups, sups, dreg)
    Emons = [getMultiUnmixedMonomials(vargroups, sups, dreg - ds[i, :]') for i = 1:s]
    Emons = [[getMultiUnmixedMonomials(vargroups, sups, dreg - ones(Int, length(dreg))')]; Emons]
    A₀mons = getMultiUnmixedMonomials(vargroups, sups, ones(Int, length(dreg))')
    D = montoexp(Dmons)
    E = [montoexp(EE) for EE ∈ Emons]
    A₀ = montoexp(A₀mons)
    return A₀, E, D
end

# Solve a square, dense system of equations.
# This uses the DBD method to compute the cokernel if DBD = true.
function solve_CI_dense(f, x; DBD = true, complex = false, check_criterion = false, verbose::Bool=false)
    if DBD
        n = length(x)
        s = n
        ds = fill(0, s)
        for i = 1:s
            ds[i] = maximum(degree.(terms(f[i])))
        end
        dreg = sum(ds) - n + 1
        cokernel, support, topdeg = getKernel_dense_DBD(f, x, dreg; complex = true, verbose = verbose)
        A₀ = montoexp(getMonomials(x, 1))
        extdeg = topdeg .- ds
        extdeg = [topdeg - 1; extdeg]
        E = [montoexp(getMonomials(x, extdeg[i])) for i = 1:length(f)+1]
        D = montoexp(getMonomials(x, topdeg))
        solve_EV(
            f,
            x,
            A₀,
            E,
            D;
            complex = complex,
            N = cokernel,
            check_criterion = check_criterion,
            schurfac = true,
            verbose = verbose,
        )
    else
        (A₀, E, D) = get_AT_CI_dense(f, x)
        solve_EV(
            f,
            x,
            A₀,
            E,
            D;
            complex = complex,
            check_criterion = check_criterion,
            schurfac = true,
            verbose = verbose,
        )
    end
end

# Solve a square, multi-graded dense system of equations.
function solve_CI_multi_dense(f, vargroups, ds; complex = false, check_criterion = false, verbose::Bool=false)
    (A₀, E, D) = get_AT_CI_multi_dense(f, vargroups,ds)
    solve_EV(
        f,
        vcat(vargroups...),
        A₀,
        E,
        D;
        complex = complex,
        check_criterion = check_criterion,
        schurfac = true,
        verbose = verbose,
    )
end

# Solve a square, multi-unmixed system of equations.
function solve_CI_multi_unmixed(f, vargroups, sups, ds; AT = nothing, complex = false, check_criterion = false, verbose::Bool=false)
    if isnothing(AT)
        (A₀, E, D) = get_AT_CI_multi_unmixed(f, vargroups, sups, ds)
    else
        A₀ = AT[1]
        E = AT[2]
        D = AT[3]
    end
    sol = solve_EV(
        f,
        vcat(vargroups...),
        A₀,
        E,
        D;
        complex = complex,
        check_criterion = check_criterion,
        schurfac = true,
        verbose = verbose
    )
    return sol, A₀, E, D
end

# Generate a random dense system of equations of degrees specified by ds.
function getRandomSystem_dense(x, ds; complex = false)
    n = length(x)
    s = length(ds)
    M = [getMonomials(x, d) for d ∈ ds]
    if complex
        f = [randn(ComplexF64, length(MM))' * MM for MM ∈ M]
    else
        f = [randn(length(MM))' * MM for MM ∈ M]
    end
    return f[:]
end

# Generate a random multi-graded dense system of equations. The r variable groups
# are given by vargroups, and the multidegrees in the (s x r) array ds.
function getRandomSystem_multi_dense(vargroups, ds; complex = false)
    M = [getMultiMonomials(vargroups, ds[i, :]) for i = 1:size(ds, 1)]
    if complex
        f = [randn(ComplexF64, length(MM))' * MM for MM ∈ M]
    else
        f = [randn(length(MM))' * MM for MM ∈ M]
    end
    return f[:]
end

# Generate a random multi-unmixed system of equations. The r variable groups
# are given by vargroups, the corresponding polytopes are defined by the supports in sups and the multidegrees in the (s x r) array ds.
function getRandomSystem_multi_unmixed(vargroups, sups, ds; complex = false)
    M = [getMultiUnmixedMonomials(vargroups,sups,ds[i, :]) for i = 1:size(ds, 1)]
    if complex
        f = [randn(ComplexF64, length(MM))' * MM for MM ∈ M]
    else
        f = [randn(length(MM))' * MM for MM ∈ M]
    end
    return f[:]
end

# Compute all monomials of multidegree ds in the groups of variables in vargroups.
function getMultiMonomials(vargroups, ds)
    aux = 1
    for i = 1:length(vargroups)
        aux = aux * (1 + sum(vargroups[i]))^(ds[i])
    end
    monomials(aux)
end

# Compute all monomials of multidegree ds in the groups of variables in vargroups with respect to the supports in sups.
function getMultiUnmixedMonomials(vargroups, sups, ds)
    aux = 1
    for i = 1:length(vargroups)
        aux = aux * (sum(exptomon(sups[i],vargroups[i])))^(ds[i])
    end
    P = newtonPolytope(aux,vcat(vargroups...))
    latpts = getLatticePoints(P)
    exptomon(latpts,vcat(vargroups...))
end

# Compute an admissible tuple for a square, sparse polynomial system which is generic with respect to its support,
# in the sense of "Toric Eigenvalue Methods for Solving Sparse Polynomial Systems".
function get_AT_CI_mixed(f, x)
    NP = [newtonPolytope(ff, x) for ff ∈ f]
    NP0 = newtonPolytope(1 + sum(x), x)
    NPS = [NP0; NP]
    Minksum, vtxmtx = sumAll2(NPS)
    D = getLatticePoints(Minksum)
    inds = [setdiff(1:length(NPS), i) for i = 1:length(NPS)]
    NPi = [sumAll2(NPS[ii])[1] for ii ∈ inds]
    E = [getLatticePoints(Δ) for Δ ∈ NPi]
    A₀ = getLatticePoints(NP0)
    return A₀, E, D
end

# Solve a square, sparse polynomial system which is generic with respect to its support,
# in the sense of "Toric Eigenvalue Methods for Solving Sparse Polynomial Systems".
function solve_CI_mixed(f, x; AT = nothing, complex = false, verbose::Bool=false)
    if isnothing(AT)
        (A₀, E, D) = get_AT_CI_mixed(f, x)
    else
        A₀ = AT[1]
        E = AT[2]
        D = AT[3]
    end
    sol = solve_EV(f, x, A₀, E, D; complex = complex, verbose = verbose)
    return sol, A₀, E, D
end

# Compute an admissible tuple for a square, sparse, unmixed polynomial system which is generic with respect to its support,
# in the sense of "Toric Eigenvalue Methods for Solving Sparse Polynomial Systems".
function get_AT_CI_unmixed(x, A, α)
    α = [1; α]
    r = getCodegree(A)
    inds = [setdiff(1:length(α), i) for i = 1:length(α)]
    Eholes = [(sum(α[ii]) - r + 1) * A for ii ∈ inds]
    Dholes = (sum(α) - r + 1) * A
    Epols = [sum([prod(x .^ (AA[i, :])) for i = 1:size(AA, 1)]) for AA ∈ Eholes]
    Dpol = sum([prod(x .^ (Dholes[i, :])) for i = 1:size(Dholes, 1)])
    Epolytopes = [newtonPolytope(ff, x) for ff ∈ Epols]
    Dpolytope = newtonPolytope(Dpol, x)
    E = [getLatticePoints(P) for P ∈ Epolytopes]
    D = getLatticePoints(Dpolytope)
    A₀pol = sum([prod(x .^ (A[i, :])) for i = 1:size(A, 1)])
    A₀polytope = newtonPolytope(A₀pol, x)
    A₀ = getLatticePoints(A₀polytope)
    return A₀, E, D
end

# Generate a random sparse unmixed system such that fᵢ has support in the convex hull of αᵢA.
function getRandomSystem_unmixed(x, A, α; complex = false)
    n = length(x)
    s = length(α)
    monsholes = [exptomon(a * A, x) for a ∈ α]
    pols = [sum(mons) for mons ∈ monsholes]
    exps = [getLatticePoints(newtonPolytope(ff, x)) for ff ∈ pols]
    M = [exptomon(e, x) for e ∈ exps]
    if complex
        f = [randn(ComplexF64, length(MM))' * MM for MM ∈ M]
    else
        f = [randn(length(MM))' * MM for MM ∈ M]
    end
    return f[:]
end

# Solve a square, sparse, unmixed polynomial system which is generic with respect to its support,
# in the sense of "Toric Eigenvalue Methods for Solving Sparse Polynomial Systems".
# !!!!!!!!!!!!!!!!!!! MAKE SURE A CONTAINS 0
function solve_CI_unmixed(f, x, A, α; AT = nothing, verbose::Bool=false)
    if isnothing(AT)
        (A₀, E, D) = get_AT_CI_unmixed(x, A, α)
    else
        A₀ = AT[1]
        E = AT[2]
        D = AT[3]
    end
    sol = solve_EV(f, x, A₀, E, D; verbose = verbose)
    return sol, A₀, E, D
end

# Solve an overdetermined system of dense equations.
function solve_OD_dense(f, x; maxdeg = 100, complex = false, verbose::Bool=false, trySemiregular::Bool=false)
    verbose && println("Looking for an admissible tuple")
    cokernel, support, chck, topdeg, AT =
        findAdmissibleTuple_dense(f, x, maxdeg; complex = true, verbose = verbose, trySemiregular = trySemiregular)
    (A₀, E, D) = AT
    sol = solve_EV(f, x, A₀, E, D; N = cokernel, complex = complex, check_criterion = false, verbose = verbose)
    return sol
end

# Solve an overdetermined system of sparse unmixed equations.
function solve_OD_unmixed(f, x, A, α; maxdeg = 100, complex = false, verbose::Bool=false)
    verbose && println("Looking for an admissible tuple")
    cokernel, support, chck, topdeg, AT =
        findAdmissibleTuple_unmixed(f, x, A, α, maxdeg; complex = true, verbose = verbose)
    (A₀, E, D) = AT
    sol = solve_EV(f, x, A₀, E, D; N = cokernel, complex = complex, check_criterion = false, verbose = verbose)
    return sol
end

# solve vA = λvB for m x n matrices A, B with m <= n.
function solveREP(A, B; rtol = 1e-10)
    (m, n) = size(A) # should be the same as size(B)
    if m == 1
        svdobj = svd([A; B])
        if svdobj.S[2] < rtol * svdobj.S[1]
            return [mean(A ./ B)], [[1]]
        else
            return [], []
        end
    else
        O = randn(n, m)
        qrobj = qr(O)
        O = qrobj.Q[:, 1:m]
        eigenobj = eigen(transpose(A * O), transpose(B * O))
        Λ = eigenobj.values
        eigenvals = fill(0.0 + 0.0 * im, 0)
        eigenspaces = fill(fill(0.0 + 0.0 * im, m, 1), 0)
        for λ ∈ Λ
            ns = nullspace(transpose(A - λ * B), rtol = rtol)
            if size(ns, 2) > 0
                push!(eigenvals, λ)
                push!(eigenspaces, ns')
            end
        end
        return eigenvals, eigenspaces
    end
end

# Return all monomials of degree ≤ d in x (equality if homogeneous = true).
function getMonomials(x, d; homogeneous = false)
    if homogeneous
        return monomials(sum(x)^d)
    else
        return monomials((1 + sum(x))^d)
    end
end

# Return a basis for the space of vanishing polynomials in x supported on mons and vanishing on the points in pts.
function getVanishingPolynomials(pts, mons, x; augm_prec = false)
    if augm_prec
        pts = convert.(Array{Complex{BigFloat},1}, pts)
    end
    exps = montoexp(mons)
    Vdm = [[prod(pt .^ (exps[i, :])) for i = 1:size(exps, 1)] for pt ∈ pts]
    Vdm = [v / norm(v) for v ∈ Vdm]
    Vdm = hcat(Vdm...)
    N = nullspace(transpose(Vdm))
    N = convert.(Complex{Float64}, N)
    κ = cond(Vdm)
    return transpose(N) * mons, κ
end

# Return the degrees of the polynomials in f.
function getDegrees(f)
    d = fill(0, s)
    for i = 1:s
        d[i] = maximum(degree.(terms(f[i])))
    end
    return d
end

# This computes the relative backward errors of the points in sol for the system f in the variables x.
# It uses the definition in Appendix C of my thesis.
function get_residual(f, sol, x)
    n = length(x)
    residuals = zeros(length(sol))
    for k = 1:length(sol)
        ressol = 0
        for i = 1:length(f)
            R = f[i]
            T = terms(R)
            l = length(T)
            Rabs = sum(abs.([T[s](x => abs.(sol[k])) for s = 1:l]))
            R = R(x => sol[k]) / (1 + Rabs)
            ressol = ressol + abs(R)
        end
        residuals[k] = ressol / length(f)
    end
    return residuals
end

# simultaneous diagonalization of the matrices in M via Schur factorization of a random linear combination.
# returns a matrix with the eigenvalues (`third mode` of the CPD)
function simulDiag_schur_mtx(M; clustering = false, tol = 1e-4, complex = false)
    n = length(M)
    δ = size(M[1], 1)
    Mₕ = sum(randn(n) .* M)
    F = schur(Mₕ)
    if complex == false
        F = triangularize(F)
    end
    solMatrix = fill(0.0 + 0.0 * im, δ, n)
    if clustering
        oF = ordschur(F, fill(true, δ))
        v = diag(oF.Schur)
        clusters, clusvec = getClusters(v; tol = tol)
        push!(clusvec, [δ + 1])
        Q = oF.Z
        for i = 1:n
            Uᵢ = Q' * M[i] * Q
            eᵢ = fill(zero(ComplexF64), δ)
            for j = 1:length(clusvec)-1
                i₀ = clusvec[j][1]
                i₁ = clusvec[j+1][1] - 1
                ev = tr(Uᵢ[i₀:i₁, i₀:i₁]) / (i₁ - i₀ + 1)
                eᵢ[i₀:i₁] .= ev
            end
            solMatrix[:, i] = eᵢ
        end
    else
        Q = F.Z
        for i = 1:n
            Uᵢ = Q' * M[i] * Q
            if i == 1 && norm(Uᵢ - triu(Uᵢ)) > 1e-10 * norm(Uᵢ)
                println("there may be singular solutions, try the option clustering = true")
            end
            solMatrix[:, i] = diag(Uᵢ)
        end
    end
    return solMatrix
end

# returns a vector clusters such that clusters[i] contains the entries of v in the i-th cluster,
# and a vector clusvec such that v[clusvec[i]] = clusters[i].
# v is a vector of complex number which is assumed to be ordered, e.g. coming from an ordered Schur factorization
function getClusters(v; tol = 1e-4)
    clusters = [[v[1]]]
    clusvec = [[1]]
    k = 2
    while k <= length(v)
        centers = [mean(clus) for clus ∈ clusters]
        distances = [abs(v[k] - c) for c ∈ centers]
        if minimum(distances) < tol
            ind = argmin(distances)
            push!(clusters[ind],v[k])
            push!(clusvec[ind],k)
        else
            push!(clusvec,[k])
            push!(clusters,[v[k]])
        end
        k = k + 1
    end
    return clusters, clusvec
end

end
