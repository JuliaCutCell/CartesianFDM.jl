none() = nothing
none(::NTuple{N}) where {N} = ntuple(i -> nothing, N)

function scalarfield(sym::Symbol, n, args...; mask=none(), tag=cell(n))
    data = @variables($sym[Base.OneTo(prod(n))]) |> only |> collect
    apply!(data, mask, n, args...)
    TaggedVector(data, tag)
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

