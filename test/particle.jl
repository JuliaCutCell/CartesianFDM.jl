using CartesianFDM
using LinearAlgebra
using SparseArrays

n = (9, 8)
topo = (nonperiodic(), periodic())

ops = fdmoperators(topo, n)

# metrics
V = map(mask(ops, ones(Float64, prod(n)))) do x
    iszero(x) ? eps(x) : x
end
A = mask(ops, ntuple(length(n)) do i
                  ones(Bool, prod(n))
              end)

#=
using JLD

data = load("particle.jld")
V = data["V"]
A = data["A"]
=#

# primary
T = scalar(:T, n)
D = ones(Float64, prod(n))

# secondary
G = gradient(dir, ops, A, V, T, D)
L = divergence(dir, ops, A, G)

### potential flow
Φ = scalar(:Φ, n)

# dirichlet = true
bc = Mixed([reshape([i > n[1] - 5 for i in 1:n[1], j in 1:n[2]], :),
            ones(Bool, prod(n))])
U = gradient(bc, ops, A, V, 1, Φ, 0, -1)
Δ = map(enumerate(divergence(bc, ops, A, U, 1))) do (i, el)
    iszero(el) ? -8Φ[i] : el
end

potential = eval(first(build_function(Δ, Φ)))

using NLsolve

sol = nlsolve(zeros(prod(n))) do res, Φ
    res .= potential(Φ)
end
phi = getproperty(sol, :zero)

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

