struct BackwardArray{F,T,N,AA<:AbstractArray{T,N}} <: AbstractArray{T,N}
    func::F
    dim::Int
    data::AA
end

parent(arr::BackwardArray) = arr.data
size(arr::BackwardArray) = size(parent(arr))

function getindex(arr::BackwardArray, index::Int...)
    (; func, dim, data) = arr
    if index[dim] != first(axes(data, dim))
        shifted = map(enumerate(index)) do (d, i)
            i - (d == dim)
        end
        x = getindex(data, shifted...)
    else
        x = zero(eltype(data))
    end
    y = getindex(data, index...)
    func(y, x)
end

struct ForwardArray{F,T,N,AA<:AbstractArray{T,N}} <: AbstractArray{T,N}
    func::F
    dim::Int
    data::AA
end

parent(arr::ForwardArray) = arr.data
size(arr::ForwardArray) = size(parent(arr))

function getindex(arr::ForwardArray, index::Int...)
    (; func, dim, data) = arr
    x = getindex(data, index...)
    if index[dim] != last(axes(data, dim))
        shifted = map(enumerate(index)) do (d, i)
            i + (d == dim)
        end
        y = getindex(data, shifted...)
    else
        y = zero(eltype(data))
    end
    func(y, x)
end

struct MaskedArray{I,T,N,AA<:AbstractArray{T,N}} <: AbstractArray{T,N}
    indices::I
    data::AA
end

parent(arr::MaskedArray) = arr.data
size(arr::MaskedArray) = size(parent(arr))

function getindex(arr::MaskedArray, index::Int...)
    (; indices, data) = arr
    CartesianIndex(index) in indices ? getindex(data, index...) : zero(eltype(data))
end

σ(y, x) = y + x
σ⁻(dim, data) = BackwardArray(σ, dim, data)
σ⁺(dim, data) = ForwardArray(σ, dim, data)

δ(y, x) = y - x
δ⁻(dim, data) = BackwardArray(δ, dim, data)
δ⁺(dim, data) = ForwardArray(δ, dim, data)

function μ(dim, data)
    indices = ntuple(ndims(data)) do d
        iter = axes(data, d)
        start = iter[begin + (d == dim)]
        stop = iter[end - 1]
        UnitRange(start, stop)
    end |> CartesianIndices
    MaskedArray(indices, data)
end

