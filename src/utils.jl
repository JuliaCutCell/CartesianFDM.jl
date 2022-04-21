#=
struct Eye{N} end

(::Eye{N})(x...) where {N} = x[N]

const id = Eye{1}()

allequal(iter) = all(==(first(iter)), Base.tail(iter))
=#
(f::Fix1)(y...) = f.f(f.x, y...)

#=
struct Fix1{F,T} <: Function
    f::F
    x::T

    Fix1(f::F, x::T) where {F,T} = new{F,T}(f, x)
    Fix1(f::Type{F}, x::T) where {F,T} = new{Type{F},T}(f, x)
end
=#

struct Stack{F} <: Function
    fs::F
end

Stack(fs...) = Stack(fs)

(f::Stack)(x) = x .|> f.fs

###
struct Splat{F} <: Function
    f::F

    Splat(f::F) where {F} = new{F}(f)
    Splat(f::Type{F}) where {F} = new{Type{F}}(f)
end

(f::Splat{F})(x) where {F} = f.f(x...)

#
#stack(fs...) = x -> x .|> fs

eachdim(x) = eachindex(x)
eachdim(::NTuple{N}) where {N} = ntuple(identity, Val(N))

flatten() = ()
flatten(x::NTuple, args...) = x..., flatten(args...)...

