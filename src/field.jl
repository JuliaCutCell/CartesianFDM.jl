# TaggedArray
# Generalize this to TaggedArray{T,N}
# No need to overspecify tag type
struct Field{T,N,V<:AbstractVector{T}} <: AbstractVector{T}
    data::V
    tag::NTuple{N,Bool}
end

staggering(vec::Field) = vec.tag

parent(vec::Field) = vec.data
size(vec::Field) = size(parent(vec))
getindex(vec::Field, i...) = getindex(parent(vec), i...)
setindex!(vec::Field, val, i...) = setindex!(parent(vec), val, i...)

none() = nothing
none(::NTuple{N}) where {N} = ntuple(i -> nothing, N)

function scalarfield(sym::Symbol, n, args...; mask=none(), tag=cell(n))
    data = @variables($sym[Base.OneTo(prod(n))]) |> only |> collect
    apply!(data, mask, n, args...)
    Field(data, tag)
end

function vectorfield(sym::Symbol, n, args...; mask=none(n), tag=face(n))
    broadcast(components(sym, n), mask, tag) do s, m, t
        scalarfield(s, n, args..., mask=m, tag=t)
    end
end

apply!(_, ::Nothing, _...) = nothing

function apply!(data, mask, n, top, val=zero(eltype(data)))
    for (i, (t, (left, right))) in enumerate(zip(top, mask))
        isperiodic(t) && continue
        arr = reshape(data, prod(n[begin:i-1]), n[i], prod(n[i+1:end]))
        left && (arr[:, begin, :] .= val)
        right && (arr[:, end, :] .= val)
    end
end

#=
function mask!(vec::Field, n, top, val=zero(eltype(vec)))
    (;tag, data) = vec
    data = reshape(data, n...)
    for (i, el) in enumerate(zip(n, top, tag))
        mask!(data, i, el..., val)
    end
    vec
end

mask!(data, dir, n, ::Periodic, ::Vararg) = nothing

function mask!(data, dir, n, ::NonPeriodic, tag, val)
    if tag
        rng = map(enumerate(axes(data))) do (i, el)
            i == dir ? 1 : el
        end
        data[rng...] .= val
    end

    rng = map(enumerate(axes(data))) do (i, el)
        i == dir ? n : el
    end
    data[rng...] .= val

    nothing
end

=#

