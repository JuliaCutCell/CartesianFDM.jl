struct CartesianCapacities{N,V,W,A,B}
    surface::Tuple{NTuple{N,A},NTuple{N,B}}
    volume::Tuple{V,NTuple{N,W}}
end

"""
    cartesiancapacities(n)

Symbolic.

"""
function cartesiancapacities(top, n)
    surface = vector(:A, n), vector(:B, n)
    volume = scalar(:V, n), vector(:W, n)
    CartesianCapacities(surface, volume)
end

