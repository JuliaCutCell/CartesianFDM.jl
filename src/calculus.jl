transform = scalarize

"""
    gradient(A, V, T, D)

Compute the gradient of T with Dirichlet boundary condition D.

"""
function gradient(A, V, T, D)
    map(eachindex(A)) do i
        (δ⁻(i, σ⁺(i, A[i]) .* T) .- σ⁻(i, δ⁺(i, A[i]) .* D)) ./ σ⁻(i, V)
    end .|> transform
end

"""
    gradient(A, V, H, T, N)

Compute the gradient of T with Neumann boundary condition N.

"""
function gradient(A, V, H, T, N)
    map(eachindex(A)) do i
        (A[i] .* δ⁻(i, T) - σ⁻(i, δ⁺(i, A[i]) .* H .* N)) ./ σ⁻(i, V)
    end .|> transform
end

"""
    divergence(A, U)

Compute the volume-weighted divergence of U with Dirichlet boundary conditions.

"""
function divergence(A, U)
    mapreduce(+, eachindex(U)) do i
        σ⁺(i, A[i]) .* δ⁺(i, U[i])
    end |> transform
end

"""
    divergence(A, U, W)

Compute the volume-weighted divergence of U with Neumann boundary conditions.

"""
function divergence(A, U, W)
    mapreduce(+, eachindex(U)) do i
        δ⁺(i, A[i] .* U[i]) .- δ⁺(i, A[i]) .* W
    end |> transform
end

