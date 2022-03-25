using CartesianFDM

n = (5, 5)
top = ntuple(periodic, length(n))

Φ = scalarfield(:Φ, n)
U = vectorfield(:U, n)

(; θ, δ, σ, ω) = nonlocaloperators(top, n)

# check permanent product identity
for (f, g, h, u) in zip(δ, σ, ω, U)
    id = Φ .* (f(u .* g(Φ)) .+ g(u .* f(Φ))) .- f(u .* h(Φ, Φ))
    @assert all(iszero, expand.(id))

    id = u .* (f(Φ .* g(u)) .+ g(Φ .* f(u))) .- f(Φ .* h(u, u))
    @assert all(iszero, expand.(id))
end

# first order upwind
upwind((left, right), u) =
    max(u, zero(u)) * left + min(u, zero(u)) * right

upwindop = CompositeOperator.(tuple.(θ, Ref(identity)), Ref(upwind))

conv = mapreduce(+, δ, upwindop, U) do d, op, u
    d(op(Φ, u))
end

