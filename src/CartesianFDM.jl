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
export NonPeriodic, nonperiodic

export Dirichlet, dir
export Neumann, neu
export Mixed

export FDMOperators, fdmoperators

export mask, gradient, divergence, strainrate, divergence2, dissipation, permanent

include("symbolics.jl")
include("topology.jl")
include("boundary.jl")
include("operators.jl")
include("calculus.jl")

end
