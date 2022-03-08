components(sym::Symbol, n) =
    map(Base.OneTo(n)) do i
        Symbol(sym, subscripts[i])
    end

scalar(sym::Symbol, n::TupleN{Int}) =
    @variables($sym[Base.OneTo.(n)...]) |> only

scalar(sym::Symbol, arr::AbstractArray) =
    @variables($sym[axes(arr)...]) |> only

vector(sym::Symbol, n::TupleN{Int}) =
    scalar.(components(sym, length(n)), Ref(n))

vector(sym::Symbol, arr::TupleN{AbstractArray}) =
    scalar.(components(sym, length(arr)), arr)

vector(sym::Symbol, arr::AbstractArray) =
    scalar.(components(sym, ndims(arr)), Ref(arr))

@register_symbolic δ(x::Real, y::Real)::Real
@register_symbolic σ(x::Real, y::Real)::Real

