using CartesianFDM

n = (17, 17)
bc = (dirichlet(), dirichlet())

ops = fdmoperators(bc, n)

### symbolic
ε = only(@variables(ε))

# metrics
V = replace!(mask(ops, scalar(:V, n))) do el
    iszero(el) ? ε : el
end
A = mask(ops, vector(:A, n))

# primary
T = scalar(:T, n)
D = scalar(:D, n)

# secondary
G = gradient(ops, A, V, T, D)
L = divergence(ops, A, G)

### numerical
V̂ = map(mask(ops, ones(Float64, prod(n)))) do x
    iszero(x) ? eps(x) : x
end
Â = mask(ops, ntuple(length(n)) do i
                  ones(Bool, prod(n))
              end)
D̂ = zeros(Bool, prod(n))

Ĝ = gradient(ops, Â, V̂, T, D̂)
L̂ = divergence(ops, Â, Ĝ)

# unsteady heat conduction on a square
# (homogeneous B.C. is non-periodic)
using Symbolics

@variables t p
(rhs, rhs!) = build_function(L̂, T, p, t)

using DifferentialEquations

tspan = (0.0, 100.0)
T̂ = ones(prod(n))
prob = ODEProblem(eval(rhs!), T̂, tspan)
sol = solve(prob)

using Plots

heatmap(reshape(sol(1), n...))
