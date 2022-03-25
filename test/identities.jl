using CartesianFDM

n = (5, 5)
top = ntuple(periodic, length(n))

Φ = scalarfield(:Φ, n)
U = vectorfield(:U, n)

(; θ, δ, σ, ω) = nonlocaloperators(top, n)

# check permanent product identity
for (Δ, Σ, Ω, u) in zip(δ, σ, ω, U)
    id = Φ .* (Δ(u .* Σ(Φ)) .+ Σ(u .* Δ(Φ))) .- Δ(u .* Ω(Φ, Φ))
    @assert all(iszero, expand.(id))

    id = u .* (Δ(Φ .* Σ(u)) .+ Σ(Φ .* Δ(u))) .- Δ(Φ .* Ω(u, u))
    @assert all(iszero, expand.(id))
end

# first order upwind
upwind((left, right), u) =
    max(u, zero(u)) * left + min(u, zero(u)) * right

upwindop = CompositeOperator.(tuple.(θ, Ref(identity)), Ref(upwind))

conv = mapreduce(+, δ, upwindop, U) do Δ, op, u
    Δ(op(Φ, u))
end

