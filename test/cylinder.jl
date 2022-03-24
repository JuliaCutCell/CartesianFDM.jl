#using Revise
using LinearAlgebra
using SparseArrays
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

# initialization
_, ∇Φ = potentialflow(ctx, A, V, N)

# mass matrix
(σ⁻, _) = getproperty(ctx, :σ)

Mᵥ = map(eachindex(σ⁻)) do i
    M = σ⁻[i] * V / 2
    map(M) do el
        abs(el) ≤ 100eps(1.0) ? prod(h) : el
    end
end
#reshape(Mᵥ[2], n...)[n[1], :] .= prod(h)

Mₚ = map(cont) do el
    iszero(el) ? prod(h) : zero(first(h))
end

M = Diagonal(vcat(Mᵥ..., Mₚ))

Re = 10.
coupled = false

if coupled
    mom = grad ./ 2 .+ visc ./ 8Re
    rhs = vcat(mom..., cont ./ 2)
    var = vcat(U..., P)

    filename = "stokes.jl"
    #(fun, fun!) = build_function(rhs, var)
    #open(filename, "w") do file
    #    write(file, string(fun))
    #    write(file, "\n")
    #    write(file, string(fun!))
    #end
    include(filename)

    using DifferentialEquations

    u₀ = vcat(∇Φ..., zeros(prod(n)))
    du₀ = zero(u₀)
    tspan = (0.0, 1.0)

    ### option 2
    diffvar = (!iszero).(vcat(Mᵥ..., Mₚ))

    function dae(out, du, u, M, t)
        stokes!(out, u, nothing, t)
        out .= out - M * du
        nothing
    end

    prob = DAEProblem(dae, du₀, u₀, tspan, M, differential_vars=diffvar)

    using Sundials

    integrator = init(prob, IDA())
    sol = step!(integrator)

    ### option 1
    f = ODEFunction(stokes!, mass_matrix=M)
    prob = ODEProblem(f, u₀, tspan)

    integrator=init(prob, Rodas5(), reltol=1e-3, abstol=1e-3)
    step!(integrator)
end

# pressure projection
#    mom = grad ./ 2 .+ visc ./ 8Re
τ = 0.001
mom = visc ./ 8Re

## setup
### prediction
M = vcat(Mᵥ...)

J = Symbolics.jacobian(vcat(mom...), vcat(U...)) 

mat = zeros(2prod(n), 2prod(n))
broadcast!(mat, J) do el
    Float64(el.val)
end
mat = Diagonal(M) .- τ .* sparse(mat)

### projection
#u^n+1 = u⋆ - ∇(τP)
#∇⋅u⋆ = Δ(τP)

∇P = gradient(neu, ctx, A, 0, P, 0)
∇P = map(Mᵥ, ∇P) do M, G
    G ./ M
end
ΔP = divergence(neu, ctx, A, ∇P, 0)

J = Symbolics.jacobian(ΔP, P) 

lap = zeros(prod(n), prod(n))
broadcast!(lap, J) do el
    Float64(el.val)
end
lap = sparse(lap)

#=
niter = 100

sol = vcat(∇Φ...)
for k in 1:niter
    E₀ = strainrate(dir, ctx, A, V, sol, W)
    visc₀ = vcat(divergence(dir, ctx, A, E₀)...)

    b = τ .* visc₀ ./ 8Re
    sol .+= mat \ b

    m = prod(n)

    # poisson
    rhs = divergence(neu, ctx, A, [sol[1:m], sol[1+m:2m]], N)
grad = -gradient(neu, ctx, A, 0, P, 0)
end

using Plots

heatmap(reshape(sol[1:prod(n)], n...))
=#


# test divergence
#=
V̂ = map(mask(ctx, fill(prod(h), prod(n)))) do x
    iszero(x) ? eps(x) : x
end
=#

#=
val = map(eachindex(h)) do i
    prod(h[j] for j in eachindex(h) if j ≠ i)
end

Â = mask(ctx, [fill(val[i], prod(n)) for i in eachindex(n)])

Ŵ = mask(ctx, ones(prod(n)))
Û = [ones(prod(n)), zeros(prod(n))]

@assert all(iszero, divergence(neu, ctx, Â, Û, Ŵ))
foo = divergence(neu, ctx, mask(ctx, vector(:A, n)), Û, 0)
bar = divergence(neu, ctx, mask(ctx, vector(:A, n)), 0, Ŵ)
=#
#=
W =
=#

