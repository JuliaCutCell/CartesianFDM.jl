using CartesianFDM
using Test

@testset "1D identities" begin
    n = (5,)
    top = (periodic(),)

    ctx = cartesianfdmcontext(top, n)

    # assumes 1D
    ((δ⁻,), (δ⁺,)) = getproperty(ctx, :δ)
    ((σ⁻,), (σ⁺,)) = getproperty(ctx, :σ)
    ((τ⁻,), (τ⁺,)) = getproperty(ctx, :τ)

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

@testset "star stencil" begin
    using SparseArrays

    n = (3, 4, 5)
    top = (periodic(), nonperiodic(), periodic())

    ctx = cartesianfdmcontext(top, n)
    (τ⁻, τ⁺) = getproperty(ctx, :τ)

    X = scalar(:X, n)

    ###
    F = rand(prod(n)) .* X
    for i in eachindex(top)
        F .+= τ⁻[i] * (rand(prod(n)) .* X)
    end
    for i in eachindex(top)
        F .+= τ⁺[i] * (rand(prod(n)) .* X)
    end

    ###
    sym = linearize(star, ctx, F, X)

    num = Dict{Int,Vector{Float64}}()
    for (key, val) in sym
        num[key] = eval(first(build_function(val)))()
    end

    A = spdiagm(num...)

    ###
    @test all(iszero, F .- A * X)
end

