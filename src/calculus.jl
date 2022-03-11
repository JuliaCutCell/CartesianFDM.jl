"""
    mask(ops, A)

"""
function mask(ops, V)
    (μ⁰, _) = getproperty(ops, :μ)
    μ⁰ * V
end

function mask(ops, A::Union{TupleN{T},AbstractVector{T}}) where {T<:AbstractVector}
    (_, μ¹) = getproperty(ops, :μ)
    μ¹ .* A
end

"""
    gradient(ops, A, V, T, D)

Compute the gradient of T with Dirichlet boundary condition D.

"""
function gradient(ops, A, V, T, D)
    ((δ⁻, δ⁺), (σ⁻, σ⁺)) = getproperty.(Ref(ops), (:δ, :σ))
    map(eachindex(A)) do i
        ((δ⁻[i] * ((σ⁺[i] * A[i]) .* T)) .- (σ⁻[i] * ((δ⁺[i] * A[i]) .* D))) ./ (σ⁻[i] * V)
    end
end

"""
    gradient(ops, A, V, H, T, N)

Compute the gradient of T with Neumann boundary condition N.

"""
function gradient(ops, A, V, H, T, N)
    ((δ⁻, δ⁺), (σ⁻, _)) = getproperty.(Ref(ops), (:δ, :σ))
    map(eachindex(A)) do i
        (A[i] .* (δ⁻[i] * T) - (σ⁻[i] * ((δ⁺[i] * A[i]) .* H .* N))) ./ (σ⁻[i] * V)
    end
end

"""
    divergence(ops, A, U)

Compute the volume-weighted divergence of U with Dirichlet boundary conditions.

!!! note "Factor 2!"

"""
function divergence(ops, A, U)
    ((_, δ⁺), (_, σ⁺)) = getproperty.(Ref(ops), (:δ, :σ))
    mapreduce(+, eachindex(U)) do i
        (σ⁺[i] * A[i]) .* (δ⁺[i] * U[i])
    end
end

"""
    divergence(ops, A, U, W)

Compute the volume-weighted divergence of U with Neumann boundary conditions.

"""
function divergence(ops, A, U, W)
    (_, δ⁺) = getproperty(ops, :δ)
    mapreduce(+, eachindex(U)) do i
        (δ⁺[i] * (A[i] .* U[i])) .- (δ⁺[i] * A[i]) .* W
    end
end

"""
    strainrate(ops, A, V, U, W)

!!! note

    For now, only diagonal terms (and extra factor 2).

"""
function strainrate(ops, A, V, U, W)
    ((δ⁻, δ⁺), (σ⁻, σ⁺)) = getproperty.(Ref(ops), (:δ, :σ))
    map(product(eachindex(A), eachindex(A))) do (i, j)
        if i == j
            ((2δ⁺[i] * (A[i] .* U[i])) .- ((δ⁺[i] * A[i]) .* (σ⁺[i] * W[i]))) ./ V
        else
            (4δ⁻[j] * ((σ⁻[i] * (σ⁺[j] * A[j])) .* U[i])
             .- (σ⁻[j] * ((σ⁻[i] * (δ⁺[j] * A[j])) .* W[i]))) ./ (σ⁻[j] * (σ⁻[i] * V))
        end
    end
end

