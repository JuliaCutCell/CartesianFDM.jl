"""

"""
function potentialflow(ctx, A, V, W)
    n = getproperty(ctx, :n)

    Φ = scalar(:Φ, n)

    U = gradient(neu, ctx, A, V, 0, Φ, 0)
    ΔΦ = divergence(neu, ctx, A, U, W)

    sym = linearize(star, ctx, ΔΦ, Φ)

    # linearity of gradient and divergence
    num = Dict{Int,Vector{Float64}}()
    for (key, val) in sym
        num[key] = getproperty.(val, Ref(:val))
    end

    Δ = spdiagm(num...)

    for i in axes(Δ, 1)
        Δ[i, i] = abs(Δ[i, i]) ≤ eps(Δ[i, i]) ? 1. : Δ[i, i]
    end

    U = gradient(neu, ctx, A, V, 0, zeros(prod(n)), 0)
    b = divergence(neu, ctx, A, U, W)

    phi = -Δ \ b

    phi, gradient(neu, ctx, A, V, 0, phi, 0)
end

