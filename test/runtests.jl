using CartesianFDM
using Test

@testset "1D identities" begin
    n = (5,)
    bc = (periodic(),)

    ops = fdmoperators(bc, n)

    # assumes 1D
    ((δ⁻,), (δ⁺,)) = getproperty(ops, :δ)
    ((σ⁻,), (σ⁺,)) = getproperty(ops, :σ)
    ((τ⁻,), (τ⁺,)) = getproperty(ops, :τ)

    Ψ, Φ = scalar(:Ψ, n), scalar(:Φ, n)

    @testset "2x identity 53" begin
        id = (σ⁺ * (δ⁻ * Φ)) .- (δ⁺ * (σ⁻ * Φ))
        @test all(iszero, Symbolics.expand.(id))

        id = (σ⁻ * (δ⁺ * Φ)) .- (δ⁻ * (σ⁺ * Φ))
        @test all(iszero, Symbolics.expand.(id))
    end

    @testset "2x identity 56" begin
        id = (σ⁺ * (Ψ .* (δ⁻ * Φ))) .+ (2Φ .* (δ⁺ * Ψ)) .- (δ⁺ * (Ψ .* (σ⁻ * Φ)))
        @test all(iszero, expand.(id))

        id = (σ⁻ * (Ψ .* (δ⁺ * Φ))) .+ (2Φ .* (δ⁻ * Ψ)) .- (δ⁻ * (Ψ .* (σ⁺ * Φ)))
        @test all(iszero, expand.(id))
    end

    @testset "2x identity 57" begin
        id = (Φ .* ((δ⁺ * (Ψ .* (σ⁻ * Φ))) .+ (σ⁺ * (Ψ .* (δ⁻ * Φ))))) .-
            (δ⁺ * (Ψ .* ((Φ .* (τ⁻ * Φ)) .+ ((τ⁻ * Φ) .* Φ))))
        @test all(iszero, expand.(id))

        id = (Φ .* ((δ⁻ * (Ψ .* (σ⁺ * Φ))) .+ (σ⁻ * (Ψ .* (δ⁺ * Φ))))) .-
            (δ⁻ * (Ψ .* ((Φ .* (τ⁺ * Φ)) .+ ((τ⁺ * Φ) .* Φ))))
        @test all(iszero, expand.(id))
    end

    @testset "2x identity 59" begin
        id = (((σ⁺ * Ψ) .* (δ⁺ * Φ)) .+ ((σ⁺ * Φ) .* (δ⁺ * Ψ))) - (2δ⁺ * (Ψ .* Φ))
        @test all(iszero, expand.(id))

        id = (((σ⁻ * Ψ) .* (δ⁻ * Φ)) .+ ((σ⁻ * Φ) .* (δ⁻ * Ψ))) - (2δ⁻ * (Ψ .* Φ))
        @test all(iszero, expand.(id))
    end
end

