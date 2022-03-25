struct TaggedArray{T,N,X,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    tag::X
end

const TaggedVector{T,X,A} = TaggedArray{T,1,X,A}

TaggedVector(data, tag) = TaggedArray(data, tag)

tag(x::TaggedArray) = x.tag

function tag(x::TaggedArray, ys::TaggedArray...)
    @assert all(==(tag(x)), tag.(ys))
    tag(x)
end

parent(x::TaggedArray) = x.data
size(x::TaggedArray) = size(parent(x))
getindex(x::TaggedArray, i...) = getindex(parent(x), i...)
setindex!(x::TaggedArray, val, i...) = setindex!(parent(x), val, i...)

showarg(io::IO, x::TaggedArray, toplevel) =
    print(io, typeof(x), " with tag '", tag(x), "'")

(+)(xs::TaggedArray...) = TaggedArray((+)(parent.(xs)...), tag(xs...))
(-)(xs::TaggedArray...) = TaggedArray((-)(parent.(xs)...), tag(xs...))

Base.BroadcastStyle(::Type{<:TaggedArray}) = Broadcast.ArrayStyle{TaggedArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TaggedArray}},
                      ::Type{ElType}) where ElType
    xs = findtaggedarrays(bc)
    TaggedArray(similar(Array{ElType}, axes(bc)), tag(xs...))
end

findtaggedarrays(bc::Base.Broadcast.Broadcasted) = findtaggedarrays(bc.args...)
findtaggedarrays(ext::Base.Broadcast.Extruded) = findtaggedarrays(ext.x)
findtaggedarrays(a::TaggedArray) = (a,)
findtaggedarrays(a) = ()
findtaggedarrays(x, xs...) = findtaggedarrays(x)..., findtaggedarrays(xs...)...

