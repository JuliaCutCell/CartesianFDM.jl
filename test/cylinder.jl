using CartesianFDM

n = (33, 32)
top = (nonperiodic(), periodic())

ctx = cartesianfdmcontext(top, n)

h = spacing.(top, n)
x = coordinate.(top, n)

cylinder(x, y, z...) = 0.25 - (2x - 1) ^ 2 - (2y - 1) ^ 2

using Vofinit

h₀ = Cdouble.((h..., sum(h) / 2))
xₑ = Cdouble[0, 0, 0, 0]

using Base.Iterators

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

