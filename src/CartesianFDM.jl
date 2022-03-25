module CartesianFDM

import Base: parent, size, getindex, setindex!, *

using Base.Iterators

using LinearAlgebra
using SparseArrays

using Reexport
@reexport using Symbolics

using StaticArrays
import StaticArrays: sacollect

const subscripts = SVector('\u2081', '\u2082', '\u2083')
const TupleN{T,N} = NTuple{N,T}

export scalar, vector

export Periodic, periodic
export NonPeriodic, nonperiodic
export cell, face

export Field
export scalarfield, vectorfield
export mask!

export operators
#export CenteredOperators, centeredoperators

export Dirichlet, dir
export Neumann, neu
export Mixed

export star

export CartesianFDMContext, cartesianfdmcontext

export CartesianCapacities, cartesiancapacities

export mask, gradient, divergence, strainrate, divergence2, dissipation, permanent

export linearize

export spacing, coordinate

export potentialflow

include("utils.jl")
include("symbolics.jl")
include("topology.jl")
include("field.jl")
include("operator.jl")
include("boundary.jl")
include("stencil.jl")
include("operators.jl")
include("capacities.jl")
include("calculus.jl")
include("linearization.jl")
include("mesh.jl")
include("flow.jl")

end
