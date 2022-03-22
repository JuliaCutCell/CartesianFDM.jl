using CartesianFDM
using LinearAlgebra
using SparseArrays

n = (33, 32)
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

# area
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

# potential flow
W = reshape(map(CartesianIndices(n)) do ind
                ind[1] ≤ 2 || ind[1] ≥ n[1] - 3 ? 1. : 0.
            end, :)
_, U = potentialflow(ctx, A, V, W)

using Plots

heatmap(reshape(U[1], n...))

#=
T = scalar(:T, n)
D = ones(prod(n))

# secondary
G = gradient(dir, ctx, A, V, T, D)
L = divergence(dir, ctx, A, G)

#
M = map(V) do el
    el ≤ 10eps(el) ? prod(h) : el
end
=#

filename = "heat.jl"

# if file already exists
#include(filename)
# otherwise
#(heat, heat!) = build_function(L ./ 2M, T)
#open(filename, "w") do file
#    write(file, string(heat))
#    write(file, "\n")
#    write(file, string(heat!))
#end

#=
using DifferentialEquations

T₀ = zeros(prod(n))
tspan = (0.0, 1.0)
prob = ODEProblem(heat, T₀, tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

using Plots

heatmap(reshape(sol(0.1), n...))
=#

#=
potential = eval(first(build_function(Δ, Φ)))

using NLsolve

sol = nlsolve(zeros(prod(n))) do res, Φ
    res .= potential(Φ)
end
phi = getproperty(sol, :zero)
=#

#J = Symbolics.jacobian(Δ, Φ)
#J = eval(first(build_function(sparse(Symbolics.jacobian(Δ, Φ)))))()

#phi = rand(prod(n))
#res1 = eval(potential)(phi)
#res2 = J * phi + eval(potential)(zeros(prod(n)))

#@assert norm(res1 - res2) < 1e-14

# also in CYLINDER!
#=
vec = diag(J)
broadcast!(vec, vec) do x
    iszero(x) ? -8one(x) : zero(x)
end

J .+= vec

phi = (-J) \ eval(potential)(zeros(prod(n)))
=#

#=
@variables t p
(rhs, rhs!) = build_function(L, T, p, t)

using DifferentialEquations

tspan = (0.0, 100.0)
T₀ = zeros(prod(n))
prob = ODEProblem(eval(rhs!), T₀, tspan)
sol = solve(prob)

using Plots

heatmap(reshape(sol(0.1), n...))
=#

### Step 1 -- thermal flow
#
# 1. Potential flow ;
# 2. Add convection ;
# 3. Neumann B. C. on the outflow ;
# 4. Robin B. C.
#
### Step 2 -- chemically reacting flow
#
# 1. More scalars with Fick's law ;
# 2. Heterogeneous reactions (with Robin B. C.) ;
# 3. Homogeneous reactions ;
# 4. Fluid-solid interaction with quasi-steady assumption.
#
### Step 2bis -- coupling with Navier-Stokes solver
#
### Step 3 -- senstivity analysis

