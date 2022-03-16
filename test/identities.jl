using CartesianFDM

n = (5,)
bc = ntuple(periodic, length(n))

ops = fdmoperators(bc, n)

ψ = scalar(:ψ, n)
φ = scalar(:φ, n)

# assumes 1d
(((δ⁻,), (δ⁺,)), ((σ⁻,), (σ⁺,))) = getproperty.(Ref(ops), (:δ, :σ))

# identity 53
id = (σ⁺ * (δ⁻ * φ)) .- (δ⁺ * (σ⁻ * φ))
@assert all(iszero, Symbolics.expand.(id))

id = (σ⁻ * (δ⁺ * φ)) .- (δ⁻ * (σ⁺ * φ))
@assert all(iszero, Symbolics.expand.(id))

# identity 56 - cell
id = (σ⁺ * (ψ .* (δ⁻ * φ))) .+ (2φ .* (δ⁺ * ψ)) .- (δ⁺ * (ψ .* (σ⁻ * φ)))
@assert all(iszero, expand.(id))

id = (σ⁻ * (ψ .* (δ⁺ * φ))) .+ (2φ .* (δ⁻ * ψ)) .- (δ⁻ * (ψ .* (σ⁺ * φ)))
@assert all(iszero, expand.(id))

# permanent product

