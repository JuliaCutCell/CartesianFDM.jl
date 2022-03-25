#=
struct Eye{N} end

(::Eye{N})(x...) where {N} = x[N]

const id = Eye{1}()

allequal(iter) = all(==(first(iter)), Base.tail(iter))
=#

