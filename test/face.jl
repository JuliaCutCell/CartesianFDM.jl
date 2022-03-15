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
U = vector(:U, n)
W = vector(:W, n)

# secondary
E = strainrate(ops, A, V, U, W)
L = divergence2(ops, A, E)

# remove ε
L = substitute.(L, Ref(Dict([ε => 0])))

###
using SparseArrays
using Base.Iterators

J = map(product(L, U)) do (Li, Uj)
    sparse(Symbolics.jacobian(Li, Uj))
end

### numerical
V̂ = map(mask(ops, ones(Float64, prod(n)))) do x
    iszero(x) ? eps(x) : x
end
Â = mask(ops, ntuple(length(n)) do i
                  ones(Bool, prod(n))
              end)
Ŵ = zero.(Â)

Ê = strainrate(ops, Â, V̂, U, Ŵ)
L̂ = divergence2(ops, Â, Ê)

Ĵ = map(product(L̂, U)) do (Li, Uj)
    sparse(Symbolics.jacobian(Li, Uj))
end

### vector heat conduction
@variables t p
(rhs, rhs!) = build_function(vcat(L̂...), vcat(U...), p, t)

using DifferentialEquations

tspan = (0.0, 100.0)
Û = ones(length(n) * prod(n))
prob = ODEProblem(eval(rhs!), Û, tspan)
sol = solve(prob)

using Plots

heatmap(first(reshape(sol(0.1), n..., 2)))

