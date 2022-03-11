module CartesianFDM

using Base.Iterators

using LinearAlgebra
using SparseArrays

using Reexport
@reexport using Symbolics

const subscripts = ("\u2081", "\u2082", "\u2083")
const TupleN{T,N} = NTuple{N,T}

export scalar, vector
export Periodic, periodic
export Dirichlet, dirichlet
export FDMOperators, fdmoperators
export mask, gradient, divergence, strainrate

include("symbolics.jl")
include("boundary.jl")
include("operators.jl")
include("calculus.jl")

end
