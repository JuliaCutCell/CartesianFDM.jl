struct LinearMap{T,A<:AbstractMatrix{T}} <: Function
    data::A
end

parent(x::LinearMap) = x.data

(x::LinearMap)(y::AbstractVector) = parent(x) * y

