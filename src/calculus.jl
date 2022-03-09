"""
    mask(ops, A)

"""
function mask(ops, A)
    _, _, _, _, μ = unpack(ops)
    μ .* A
end

"""
    gradient(ops, A, V, T, D)

Compute the gradient of T with Dirichlet boundary condition D.

"""
function gradient(ops, A, V, T, D)
    _, _, (δ⁻, δ⁺), (σ⁻, σ⁺), _ = unpack(ops)
    map(eachindex(A)) do i
        ((δ⁻[i] * ((σ⁺[i] * A[i]) .* T)) .- (σ⁻[i] * ((δ⁺[i] * A[i]) .* D))) ./ (σ⁻[i] * V)
    end
end

"""
    gradient(ops, A, V, H, T, N)

Compute the gradient of T with Neumann boundary condition N.

"""
function gradient(ops, A, V, H, T, N)
    _, _, (δ⁻, δ⁺), (σ⁻, _), _ = unpack(ops)
    map(eachindex(A)) do i
        (A[i] .* (δ⁻[i] * T) - (σ⁻[i] * ((δ⁺[i] * A[i]) .* H .* N))) ./ (σ⁻[i] * V)
    end
end

"""
    divergence(ops, A, U)

Compute the volume-weighted divergence of U with Dirichlet boundary conditions.

"""
function divergence(ops, A, U)
    _, _, (_, δ⁺), (_, σ⁺), _ = unpack(ops)
    mapreduce(+, eachindex(U)) do i
        (σ⁺[i] * A[i]) .* (δ⁺[i] * U[i])
    end
end

"""
    divergence(ops, A, U, W)

Compute the volume-weighted divergence of U with Neumann boundary conditions.

"""
function divergence(ops, A, U, W)
    _, _, (_, δ⁺), _, _ = unpack(ops)
    mapreduce(+, eachindex(U)) do i
        (δ⁺[i] * (A[i] .* U[i])) .- (δ⁺[i] * A[i]) .* W
    end
end

