using Revise
using CartesianFDM

n = (6, 6)
top = (nonperiodic(), nonperiodic())

ε = only(@variables ε)

# cell-centered volume capacity
mask = ((false, true), (false, true))
V = scalarfield(:V, n, top, ε; mask)

# face-centered volume capacities
W = vectorfield(:W, n)

# face-centered area capacities
mask = (((true, true), (false, true)),
        ((false, true), (true, true)))
A = vectorfield(:A, n, top; mask)

# cell-centered area capacities
tag = ntuple(i -> cell(n), length(n))
mask = (((false, true), (false, true)),
        ((false, true), (false, true)))
B = vectorfield(:B, n, top; mask, tag)

#
P = scalarfield(:P, n)
ops = centeredoperators(n, top)

(; δ, σ) = ops

Q = σ[1] * P

#=
tag = ntuple(i -> cell(n), length(n))
=#

#=
using StaticArrays

m = SVector(n)
vec = rand(prod(m))

for i in eachindex(m)
    arr = reshape(vec, prod(n[begin:i-1]), n[i], prod(n[i+1:end]))
    arr[:, begin, :] .= zero(eltype(arr))
    arr[:, end, :] .= zero(eltype(arr))
end

=#
