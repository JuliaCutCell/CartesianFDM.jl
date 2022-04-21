uniswitch(i) = Fix2(getindex, i)

uniflip(i) = Fix2(i) do x, j
    ntuple(length(x)) do k
        k == j ? !x[k] : x[k]
    end
end

#import Core.Compiler: return_type
# lazy
"""

Lazy construction.

"""
struct StencilArray{S,W,P,T,N,F,A<:ArrayAbstract{N}} <: AbstractArray{T,N}
    top::P
    dim::Int
    f::F
    data::A
end

#=
eltypes(arr::AbstractArray, ::Val{0}) = Tuple{eltype(arr)}
@Base.pure eltypes(A::Tuple{Vararg{<:AbstractArray}}) = Tuple{(eltype.(A))...}
=#

function StencilArray{S,W}(top, dim, f, data::AbstractArray) where {S,W}
#    infer_eltype() = Base._return_type(f, (NTuple{W,eltype(data)},))
#    T = infer_eltype()
    T = Base._return_type(f, (NTuple{W,eltype(data)},))
    StencilArray{S,W,typeof(top),T,ndims(data),typeof(f),typeof(data)}(top, dim, f, data)
end

parent(arr::StencilArray) = arr.data
size(arr::StencilArray) = size(parent(arr))

function getindex(arr::StencilArray{S,W,Periodic}, i::Vararg{Int,N}) where {S,W,N}
    (; top, dim, f, data) = arr

    n = size(data)

    ntuple(Val(W)) do s
        j = ntuple(Val(N)) do d
            d == dim ? mod(i[d] + S + s - 2, n[d]) + 1 : i[d]
        end
        data[j...]
    end |> f
end

unsplat(x::Splat) = x.f
unsplat(x) = x

StencilArray{S,W,T}(top::P, dim, f::F, data::A) where {S,W,P,T,N,F,A<:ArrayAbstract{N}} =
    StencilArray{S,W,P,T,N,F,A}(top, dim, f, data)

#=
struct ShiftedArray{S,P,T,N,F,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    top::P
    left::Int
    right::Int
    f::F
    data::A
end
=#

struct Shift{N,T,F} <: Function
    top::T
    left::Int
    right::Int
    f::F
end

Shift{N}(top, left, right, f=zero) where {N} =
    Shift{N,typeof(top),typeof(f)}(top, left, right, f)

(shift::Shift{0})(x) = x

function shiftedrange(N, start, stop)
    res = UnitRange(min(start + max(0, N), stop),
                    max(stop + min(0, N), start))
    res
end

function (shift::Shift{N,NonPeriodic})(x) where {N}
    (; top, left, right, f) = shift

    x̂ = reshape(x, left, :, right)

    y = similar(x)
    ŷ = reshape(y, left, :, right)

    start = firstindex(x̂, 2)
    stop = lastindex(x̂, 2)
    n = size(x̂, 2)

    xrng = shiftedrange(N, start, stop)
    yrng = shiftedrange(-N, start, stop)

    ŷ[:,yrng,:] .= x̂[:,xrng,:]

    yrng = shiftedrange(sign(N) * n - N, start, stop)

    ŷ[:,yrng,:] .= f(eltype(ŷ))

    y
end

function (shift::Shift{N,Periodic})(x) where {N}
    (; top, left, right, f) = shift

    x̂ = reshape(x, left, :, right)

    y = similar(x)
    ŷ = reshape(y, left, :, right)

    start = firstindex(x̂, 2)
    stop = lastindex(x̂, 2)
    n = size(x̂, 2)

    xrng = shiftedrange(N, start, stop)
    yrng = shiftedrange(-N, start, stop)

    ŷ[:,yrng,:] .= x̂[:,xrng,:]

    xrng = shiftedrange(N - sign(N) * n, start, stop)
    yrng = shiftedrange(sign(N) * n - N, start, stop)

    ŷ[:,yrng,:] .= x̂[:,xrng,:]

    y
end

"""

1. Non-local (dah!) ;
1. Single argument (univariate) ;
1. Unidimensional otherwise 2 ^ D operators (too complex). Just compose to do
multidimensional.

"""
struct StaggeredOperator{S,T,F,B} <: Function
    switch::S
    flip::T
    forward::F
    backward::B
#    f::F
end

function (op::StaggeredOperator)(x::TaggedArray)
    (; switch, flip, forward, backward) = op
    (; data, tag) = x

    res = switch(tag) ? forward(data) : backward(data)
#        f.(broadcast.(forward, ntuple(_ -> Ref(data), length(forward)))...) :
#        f.(broadcast.(backward, ntuple(_ -> Ref(data), length(backward)))...)

    TaggedVector(res, flip(tag))
end

#struct CenteredOperator{M,F} <: Function
#    maps::M
#    f::F
#end

struct CompositeOperator{U,F} <: Function
    unary::U
    f::F
end

function (op::CompositeOperator)(xs::TaggedVector...)
    (; unary, f) = op

    ys = broadcast.(unary, Ref.(xs))

    TaggedVector(f.(parent.(ys)...), tag(ys...))
end

tilde((x⁻, x⁺), (y⁻, y⁺)) = x⁻ * y⁺ + y⁻ * x⁺

unidim_bivar(dim, eye, plus, minus, f) =
    StaggeredOperator(uniswitch(dim),
                      uniflip(dim),
                      (eye, plus),
                      (minus, eye),
                      f)

combine(f, g...) = ∘(splat(Fix1(broadcast, f)), Stack(g...), Ref)

#=
# Is this really needed?
struct CompositeOperator
    pre
    post
end
=#

function nonlocaloperators(top, n)
    τ = map(eachdim(top)) do i
        Shift{+1}(top[i], prod(n[1:i-1]), prod(n[i+1:end])),
        Shift{-1}(top[i], prod(n[1:i-1]), prod(n[i+1:end]))
    end

    θ = map(eachdim(n), τ) do i, (plus, minus)
        forward = combine(tuple, identity, plus)
        backward = combine(tuple, minus, identity)
        StaggeredOperator(uniswitch(i), uniflip(i), forward, backward)
    end

    #=
    # Staggered operators
    θ = unidim_bivar.(dims, Ref(eye), τ⁺, τ⁻, Ref(SVector))
    δ = unidim_bivar.(dims, Ref(eye), τ⁺, τ⁻, Ref(-))
    σ = unidim_bivar.(dims, Ref(eye), τ⁺, τ⁻, Ref(+))

    # Composite operators
    ω = map(θ) do op
        CompositeOperator((op, op), tilde)
    end

    (; θ, δ, σ, ω)
    =#
    (; θ)
end

#UnidimOperator
#=
function size(mat::Operator)
    (; dir, forward, backward) = mat
    @assert size(forward) == size(backward)
    size(backward)
end

function (*)(mat::Operator, vec::Field)
    (; dir, forward, backward) = mat
    (; data, tag) = vec
    Field(tag[dir] ? forward * data : backward * data,
          flip(tag, dir))
end
=#

#=
struct CenteredOperators{N,D,S}
    δ::NTuple{N,D}
    σ::NTuple{N,S}
end

function centeredoperators(n, top)
    ρ = spdiagm(0 => ones(Bool, prod(n)))
    τ = forwardshiftmatrices(top, n), backwardshiftmatrices(top, n)

    δ = τ[1] .- Ref(ρ), Ref(ρ) .- τ[2]
    σ = τ[1] .+ Ref(ρ), Ref(ρ) .+ τ[2]

    dims = ntuple(identity, length(n))

    ops = map((δ, σ)) do (forward, backward)
        map(dims, forward, backward) do dir, plus, minus
            Operator(dir, plus, minus)
        end
    end

    CenteredOperators(ops...)
end
=#

