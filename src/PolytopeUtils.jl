using Polymake

function sumAll2(NPS)
    # This computes the Minkowski sum of an array of polytopes.
    vtxmtx = vertexRepresentation(NPS[1])
    for i = 2:length(NPS)
        vtxmtx2 = vertexRepresentation(NPS[i])
        C = [
            vtxmtx[i, :] + vtxmtx2[j, :] for i = 1:size(vtxmtx, 1)
            for j = 1:size(vtxmtx2, 1)
        ]
        vtxsum = hcat(ones(length(C)), vcat(C'...))
        global P = polytope.Polytope(POINTS = vtxsum)
        vtxmtx = vertexRepresentation(P)
        println(
            "computed the sum of " *
            string(i) *
            " out of " *
            string(length(NPS)) *
            " polytopes",
        )
    end
    P, vtxmtx
end

function newtonPolytope(f, y)
    # computes the Newton polytope of the polynomial f.
    trms = terms(f)
    varfinds = findall(ℓ -> ℓ ∈ variables(f), y)
    n = length(y)
    latPts = zeros(Int64, length(trms), n + 1)
    for i = 1:length(trms)
        v = zeros(n)
        v[varfinds] = transpose(exponents(trms[i]))
        latPts[i, :] = [1 v']
    end
    P = @pm polytope.Polytope(POINTS = latPts)
    return P
end

function getLatticePoints(P)
    # computes the lattice points in (a translated version of) P
    pts_α = P.LATTICE_POINTS_GENERATORS
    exps_α = convert(Array{Int64}, pts_α[1][:, 2:end])
    shift = abs.(minimum(vcat(exps_α, zeros(Int64, 1, size(exps_α, 2))), dims = 1))
    exps_α = exps_α .+ shift
    return exps_α
end

function vertexRepresentation(P)
    # Computes a matrix whose rows are the vertices of a lattice polytope P
    vtxmtx = P.VERTICES
    vtxmtx = convert(Array{Int64}, vtxmtx)
    vtxmtx = vtxmtx[:, 2:end]
    return vtxmtx
end

function getCodegree(A)
    # Compute the codegree of the convex hull of A
    r = 1
    @polyvar x[1:size(A, 2)]
    f = sum([prod(x .^ (A[i, :])) for i = 1:size(A, 1)])
    P = newtonPolytope(f, x)
    r = P.LATTICE_CODEGREE
    return r
end
