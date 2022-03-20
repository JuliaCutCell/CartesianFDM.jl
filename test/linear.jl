using CartesianFDM
using LinearAlgebra
using SparseArrays

n = (33, 32)
top = (nonperiodic(), periodic())

ops = fdmoperators(top, n)
(τ⁻, τ⁺) = getproperty(ops, :τ)

###
X = scalar(:X, n)

###
F = rand(prod(n)) .* X

for i in eachindex(top)
    F .+= τ⁻[i] * (rand(prod(n)) .* X)
end

for i in eachindex(top)
    F .+= τ⁺[i] * (rand(prod(n)) .* X)
end

###
sym = linearize(star, top, n, ops, F, X)

num = Dict{Int,Vector{Float64}}()

for (key, val) in sym
    num[key] = eval(first(build_function(val)))()
end

bar = spdiagm(num...)

###
@assert all(iszero, F .- bar * X)

