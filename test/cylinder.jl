using Revise
using CartesianFDM

n = (17, 16)
top = (nonperiodic(), periodic())

ctx = cartesianfdmcontext(top, n)

# square domain (0, 1) x (0, 1)
h = spacing.(top, n)
x = coordinate.(top, n)

# cylinder centered at (0.5, 0.5) with radius 0.25
cylinder(x, y, z...) = 0.25 - (2x - 1) ^ 2 - (2y - 1) ^ 2

# capacities
using Vofinit

h₀ = Cdouble.((h..., sum(h) / 2))
xₑ = Cdouble[0, 0, 0, 0]

using Base.Iterators

# volume
V = map(zip(CartesianIndices(n), product(x...))) do (ind, el)
    x₀ = Cdouble.((el..., 0.))
    prod(h) * getcc(cylinder, x₀, h₀, xₑ)
end

V = map(mask(ctx, reshape(V, :))) do val
    iszero(val) ? eps(val) : val
end

A₁ = map(zip(CartesianIndices(n), product(x...))) do (ind, el)
    x₀ = Cdouble.((el..., 0.))
    h[2] * getcc(x₀, h₀, xₑ) do y...
        cylinder(x₀[1], y[2])
    end
end

A₂ = map(zip(CartesianIndices(n), product(x...))) do (ind, el)
    x₀ = Cdouble.((el..., 0.))
    h[1] * getcc(x₀, h₀, xₑ) do y...
        cylinder(y[1], x₀[2])
    end
end

A = mask(ctx, [reshape(A₁, :), reshape(A₂, :)])

P = scalar(:P, n)
U = vector(:U, n)
N = reshape(map(CartesianIndices(n)) do ind
                ind[1] ≤ 4 || ind[1] ≥ n[1] - 4 ? 1. : 0.
            end, :)

# divergence and gradient
cont = divergence(neu, ctx, A, U, N)
grad = -gradient(neu, ctx, A, 0, P, 0)

@assert begin
    mapreduce(*, eachindex(U)) do i
        D = Symbolics.jacobian(cont, U[i])
        G = Symbolics.jacobian(grad[i], P)
        all(iszero, D' - G)
    end
end

W = [reshape(map(CartesianIndices(n)) do ind
                ind[1] ≤ 4 || ind[1] ≥ n[1] - 4 ? 1. : 0.
            end, :), zeros(prod(n))]

#conv =
E = strainrate(dir, ctx, A, V, U, W)
visc = divergence(dir, ctx, A, E)

# diagonal
@assert begin
    mapreduce(*, eachindex(U)) do i
        D = Symbolics.jacobian(visc[i], U[i])
        all(≤(100eps(1.)), D-D')
    end
end

# off-diagonal
@assert begin
    B₁ = Symbolics.jacobian(visc[1], U[2])
    B₂ = Symbolics.jacobian(visc[2], U[1])
    all(≤(100eps(1.)), B₁-B₂')
end

# visc =

# test divergence
#=
V̂ = map(mask(ctx, fill(prod(h), prod(n)))) do x
    iszero(x) ? eps(x) : x
end
=#

val = map(eachindex(h)) do i
    prod(h[j] for j in eachindex(h) if j ≠ i)
end

Â = mask(ctx, [fill(val[i], prod(n)) for i in eachindex(n)])

Ŵ = mask(ctx, ones(prod(n)))
Û = [ones(prod(n)), zeros(prod(n))]

@assert all(iszero, divergence(neu, ctx, Â, Û, Ŵ))
foo = divergence(neu, ctx, mask(ctx, vector(:A, n)), Û, 0)
bar = divergence(neu, ctx, mask(ctx, vector(:A, n)), 0, Ŵ)
#=
W =
=#

