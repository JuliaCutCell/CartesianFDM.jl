using CartesianFDM

n = (6, 6)
bc = ntuple(dirichlet, length(n))

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

# remove ε
L = substitute.(L, Ref(Dict([ε => 0])))

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

###
using SparseArrays

Ĵ = sparse(Symbolics.jacobian(L̂, T))
A = eval(first(build_function(Ĵ)))()

T̂ = rand(prod(n))

b̂₁ = A * T̂
b̂₂ = eval(first(build_function(L̂, T)))(T̂)

using LinearAlgebra

@show norm(b̂₁ - b̂₂)

#=
Unsteady heat conduction on a square
(homogeneous B.C. is non-periodic).

=#
@variables t p
(rhs, rhs!) = build_function(L̂, T, p, t)

using DifferentialEquations

tspan = (0.0, 100.0)
T̂ = ones(prod(n))
prob = ODEProblem(eval(rhs!), T̂, tspan)
sol = solve(prob)

using Plots

heatmap(reshape(sol(0.1), n...))

