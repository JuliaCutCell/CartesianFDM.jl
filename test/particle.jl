using CartesianFDM
using LinearAlgebra
using SparseArrays

n = (9, 8)
bc = (dirichlet(), periodic())

ops = fdmoperators(bc, n)

# metrics
V = map(mask(ops, ones(Float64, prod(n)))) do x
    iszero(x) ? eps(x) : x
end
A = mask(ops, ntuple(length(n)) do i
                  ones(Bool, prod(n))
              end)

# primary
T = scalar(:T, n)
D = ones(Float64, prod(n))

# secondary
G = gradient(ops, A, V, T, D)
L = divergence(ops, A, G)

### potential flow
Φ = scalar(:Φ, n)

# dirichlet = true
bc = [reshape([i > n[1] - 5 for i in 1:n[1], j in 1:n[2]], :),
      ones(Bool, prod(n))]
U = gradient(ops, bc, A, V, Φ, 0)
Δ = divergence(ops, bc, A, U, -1)

(potential, _) = build_function(Δ, Φ)
J = Symbolics.jacobian(Δ, Φ)

phi = rand(prod(n))
res1 = eval(potential)(phi)
res2 = J * phi + eval(potential)(zeros(prod(n)))

@assert norm(res1 - res2) < 1e-14

for i in LinearIndices(CartesianIndices(n))[end, :]
    J[i, i] = -8.
end

phi = (-J) \ eval(potential)(zeros(prod(n)))

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

