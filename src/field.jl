struct Field{T,N,V<:AbstractVector{T}} <: AbstractVector{T}
    tag::NTuple{N,Bool}
    data::V
end

staggering(vec::Field) = vec.tag

parent(vec::Field) = vec.data
size(vec::Field) = size(parent(vec))
getindex(vec::Field, i...) = getindex(parent(vec), i...)

function scalarfield(sym::Symbol, n, args...; tag=cell(n), f=id)
    data = @variables($sym[Base.OneTo(prod(n))]) |> only |> collect
    f(Field(tag, data), n, args...)
end

function vectorfield(sym::Symbol, n, args...; tag=face(n), f=id)
    broadcast(components(sym, n), tag) do s, t
        scalarfield(s, n, args..., tag=t, f=f)
    end
end

function mask!(vec::Field, n, top)
    (;tag, data) = vec
    data = reshape(data, n...)
    for (i, el) in enumerate(zip(n, top, tag))
        mask!(data, i, el...)
    end
    vec
end

mask!(data, dir, n, ::Periodic, tag) = nothing

function mask!(data, dir, n, ::NonPeriodic, tag)
    if tag
        rng = map(enumerate(axes(data))) do (i, el)
            i == dir ? 1 : el
        end
        data[rng...] .= zero(eltype(data))
    end

    rng = map(enumerate(axes(data))) do (i, el)
        i == dir ? n : el
    end
    data[rng...] .= zero(eltype(data))

    nothing
end

