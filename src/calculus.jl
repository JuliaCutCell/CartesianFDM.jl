"""
    mask(ctx, A)

"""
function mask(ctx, V)
    (μ⁰, _) = getproperty(ctx, :μ)
    μ⁰ * V
end

function mask(ctx, A::Union{TupleN{T},AbstractVector{T}}) where {T<:AbstractVector}
    (_, μ¹) = getproperty(ctx, :μ)
    μ¹ .* A
end

"""
    permanent(ctx, Φ, Ψ)

!!! note "Factor 2!"


"""
function permanent(ctx, Φ, Ψ)
    (τ⁻, τ⁺) = getproperty(ctx, :τ)
    map(getproperty(ctx, :τ)) do iter
        map(iter) do el
            ((el * Φ) .* Ψ) .+ (Φ .* (el * Ψ))
        end
    end
end

"""
    gradient(ctx, A, V, T, D)

Compute the gradient of T with Dirichlet boundary condition D.

"""
function gradient(::Dirichlet, ctx, A, V, T, D)
    ((δ⁻, δ⁺), (σ⁻, σ⁺)) = getproperty.(Ref(ctx), (:δ, :σ))
    map(eachindex(A)) do i
        ((δ⁻[i] * ((σ⁺[i] * A[i]) .* T)) .- (σ⁻[i] * ((δ⁺[i] * A[i]) .* D))) ./ (σ⁻[i] * V)
    end
end

"""
    gradient(ctx, A, V, H, T, N)

Compute the gradient of T with Neumann boundary condition N.

!!! note

    Would it make sense to let N be a cell-centered vector-valued field?

"""
function gradient(::Neumann, ctx, A, V, H, T, N)
    ((δ⁻, δ⁺), (σ⁻, _)) = getproperty.(Ref(ctx), (:δ, :σ))
    map(eachindex(A)) do i
        (2A[i] .* (δ⁻[i] * T) - (σ⁻[i] * ((δ⁺[i] * A[i]) .* H .* N))) ./ (σ⁻[i] * V)
    end
end

"""
    gradient(ctx, A, V, H, T, N)

Compute (twice) the volume-weighted gradient of T with Neumann boundary condition N.

"""
function gradient(::Neumann, ctx, A, H, T, N)
    ((δ⁻, δ⁺), (σ⁻, _)) = getproperty.(Ref(ctx), (:δ, :σ))
    map(eachindex(A)) do i
        (2A[i] .* (δ⁻[i] * T) - (σ⁻[i] * ((δ⁺[i] * A[i]) .* H .* N)))
    end
end

"""
    gradient(ctx, bc, A, V, T, D)

Compute the gradient of T.

"""
#function gradient(ctx, bc, A, V, T, D)
function gradient((; info)::Mixed, ctx, A, V, H, T, D, N)
    ((δ⁻, δ⁺), (σ⁻, σ⁺)) = getproperty.(Ref(ctx), (:δ, :σ))
    map(eachindex(A)) do i
        ((δ⁻[i] * ((σ⁺[i] * A[i]) .* T)) .-
         (σ⁻[i] * ((δ⁺[i] * A[i]) .* ((info[i] .* D) .+ ((!).(info[i]) .* (T .+ (H .* N))))))) ./
        (σ⁻[i] * V)
    end
end

"""
    divergence(ctx, A, U)

Compute the volume-weighted divergence of U with Dirichlet boundary conditions.

!!! note "Factor 2!"

"""
function divergence(::Dirichlet, ctx, A, U)
    ((_, δ⁺), (_, σ⁺)) = getproperty.(Ref(ctx), (:δ, :σ))
    mapreduce(+, eachindex(U)) do i
        (σ⁺[i] * A[i]) .* (δ⁺[i] * U[i])
    end
end

"""
    divergence(bc, ctx, A, U, W)

Compute the volume-weighted divergence of U with Neumann boundary conditions.

!!! note "Factor 2!"

!!! warning

    It would make a lot of sense to let W be a cell-centered vector-valued field.

"""
function divergence(::Neumann, ctx, A, U, W)
    (_, δ⁺) = getproperty(ctx, :δ)
    mapreduce(+, eachindex(U)) do i
        (2δ⁺[i] * (A[i] .* U[i])) .- ((2δ⁺[i] * A[i]) .* W)
    end
end

"""
    divergence(ctx, bc, A, U, W)

!!! note "Factor 2!"

"""
function divergence((; info)::Mixed, ctx, A, U, W)
    ((_, δ⁺), (_, σ⁺)) = getproperty.(Ref(ctx), (:δ, :σ))
    mapreduce(+, eachindex(A)) do i
        (2δ⁺[i] * (A[i] .* U[i])) .-
        ((δ⁺[i] * A[i]) .* ((info[i] .* (σ⁺[i] * U[i])) .+ ((!).(info[i]) .* 2W)))
    end
end

"""
    strainrate(ctx, A, V, U, W)

!!! note

    For now, only diagonal terms (and extra factor 2).
    Also, Dirichlet.

"""
function strainrate(::Dirichlet, ctx, A, V, U, W)
    ((δ⁻, δ⁺), (σ⁻, σ⁺)) = getproperty.(Ref(ctx), (:δ, :σ))
    map(product(eachindex(A), eachindex(A))) do (i, j)
        if i == j
            ((2δ⁺[i] * (A[i] .* U[i])) .- ((δ⁺[i] * A[i]) .* (σ⁺[i] * W[i]))) ./ V
        else
            (2δ⁻[j] * ((σ⁻[i] * (σ⁺[j] * A[j])) .* U[i])
             .- (σ⁻[j] * ((σ⁻[i] * (δ⁺[j] * A[j])) .* W[i]))) ./ (σ⁻[j] * (σ⁻[i] * V)) .+
            (2δ⁻[i] * ((σ⁻[j] * (σ⁺[i] * A[i])) .* U[j])
             .- (σ⁻[i] * ((σ⁻[j] * (δ⁺[i] * A[i])) .* W[j]))) ./ (σ⁻[i] * (σ⁻[j] * V))
        end
    end
end

"""
    divergence(ctx, A, E)

!!! note

    Also, Dirichlet.

"""
function divergence(::Dirichlet, ctx, A, E::AbstractMatrix)
    ((δ⁻, δ⁺), (σ⁻, σ⁺)) = getproperty.(Ref(ctx), (:δ, :σ))
    map(eachindex(A)) do i
        mapreduce(+, eachindex(A)) do j
            if i == j
                4A[i] .* (δ⁻[i] * E[i, i])
            else
                (σ⁻[i] * (σ⁺[j] * A[j])) .* (δ⁺[j] * E[i, j])
            end
        end
    end
end

function dissipation(ctx, A, V, T)
    ((δ⁻, δ⁺), (σ⁻, σ⁺)) = getproperty.(Ref(ctx), (:δ, :σ))
    mapreduce(+, eachindex(A)) do i
        -(σ⁺[i] * ((δ⁻[i] * ((σ⁺[i] * A[i]) .* T)) .^ 2 ./ (σ⁻[i] * V)))
    end
end

