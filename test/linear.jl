using CartesianFDM
using LinearAlgebra
using SparseArrays

n = (33, 33)
top = ntuple(nonperiodic, length(n))

ops = fdmoperators(top, n)

X = scalar(:X, n)

###
foo = Dict{Int,typeof(X)}()

for i in eachindex(top)
    p = prod(n[1:i-1])
    m = prod(n) - p
    foo[-p] = rand(m)
end

foo[0] = rand(prod(n))

for i in eachindex(top)
    p = prod(n[1:i-1])
    m = prod(n) - p
    foo[+p] = rand(m)
end

bar = spdiagm(foo...)

###
F = bar * X

foo = linearize(star, top, n, ops, F, X)

baz = eval(first(build_function(bar)))()

###
@assert all(iszero, F .- baz * X)

