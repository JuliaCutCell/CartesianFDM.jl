components(sym::Symbol, ::NTuple{N}) where {N} =
    sacollect(SVector{N,Symbol}, Symbol(sym, subscripts[i]) for i in 1:N)

scalar(sym::Symbol, n::TupleN{Int}) =
    @variables($sym[Base.OneTo(prod(n))]) |> only |> collect

vector(sym::Symbol, n::TupleN{Int}) =
    scalar.(components(sym, n), Ref(n))

vector(sym::Symbol, arr::TupleN{AbstractArray}) =
    scalar.(components(sym, size(arr)), arr)

