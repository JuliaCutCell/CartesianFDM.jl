components(sym::Symbol, n) =
    map(Base.OneTo(n)) do i
        Symbol(sym, subscripts[i])
    end

scalar(sym::Symbol, n::TupleN{Int}) =
    @variables($sym[Base.OneTo(prod(n))]) |> only |> collect

vector(sym::Symbol, n::TupleN{Int}) =
    scalar.(components(sym, length(n)), Ref(n))

vector(sym::Symbol, arr::TupleN{AbstractArray}) =
    scalar.(components(sym, length(arr)), arr)

