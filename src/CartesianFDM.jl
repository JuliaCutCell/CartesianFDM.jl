module CartesianFDM

import Base: Fix1, Fix2, splat, parent, size, getindex, setindex!, showarg, +, -

using Base.Iterators

using LinearAlgebra
using SparseArrays

using Reexport

@reexport using Symbolics

@reexport using StaticArrays
import StaticArrays: sacollect

const ArrayAbstract{N,T} = AbstractArray{T,N}

const subscripts = SVector('\u2081', '\u2082', '\u2083')
const TupleN{T,N} = NTuple{N,T}

export Splat
export flatten, eachdim

export scalar, vector

export Periodic, periodic
export NonPeriodic, nonperiodic
export cell, face

export TaggedVector

export scalarfield, vectorfield
export mask!

export uniswitch, uniflip
export StaggeredOperator, CenteredOperator, CompositeOperator
export nonlocaloperators
export StencilArray
export Shift
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
include("array.jl")
include("field.jl")
include("operator.jl")
include("boundary.jl")
include("stencil.jl")
include("operators.jl")
include("linear.jl")
include("capacities.jl")
include("calculus.jl")
include("linearization.jl")
include("mesh.jl")
include("flow.jl")

end
