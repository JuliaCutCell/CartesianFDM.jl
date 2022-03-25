using CartesianFDM

n = (5, 5)
top = ntuple(periodic, length(n))

Φ = scalarfield(:Φ, n)
Ψ = scalarfield(:Ψ, n)#, tag=face(1, n))

(; θ, δ, σ, ω) = nonlocaloperators(top, n)
#(θ,), (δ,), (σ,), (ω,) = θ, δ, σ, ω

#=
id = Φ .* (δ(Ψ .* σ(Φ)) .+ σ(Ψ .* δ(Φ))) .- δ(Ψ .* ω(Φ, Φ))
=#

#=
findtaggedarrays(bc::Base.Broadcast.Broadcasted) = findtaggedarrays(bc.args...)
findtaggedarrays(a::TaggedArray) = (a,)
findtaggedarrays(a) = ()
findtaggedarrays(x, xs...) = findtaggedarrays(x)..., findtaggedarrays(xs...)...
=#


"""

<https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting>

"""
#=
struct FieldStyle <: Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:Field}) = FieldStyle()

FieldStyle(::Val{0}) = FieldStyle()
FieldStyle(::Val{1}) = FieldStyle()
#FieldStyle(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()

function Base.similar(bc::Broadcast.Broadcasted{FieldStyle}, ::Type{T}) where {T}
    A = find_fields(bc)
    tags = getproperty.(A, Ref(:tag))
    @assert all(==(first(tags)), Base.tail(tags))
    Field(similar(Array{T}, axes(bc)), first(tags))
end

find_fields(bc::Base.Broadcast.Broadcasted) = find_fields(bc.args...)
find_fields(a::Field, rest...) = a, find_fields(rest...)...
find_fields(a::Any, rest...) = find_fields(rest...)
find_fields() = ()

σ(Ψ) .* Φ
=#


#=
function find_field(a::Any, rest...)
    @show a, rest
    find_field(rest...)
end
find_field(::Tuple{}) = nothing
find_field() = nothing
=#

#=
find_field(args::Tuple) = find_field(find_field(args[1]), Base.tail(args))
find_field(x) = x
find_field(::Tuple{}) = nothing
find_field(a::Field, rest) = a
find_field(::Any, rest) = find_field(rest)
=#

