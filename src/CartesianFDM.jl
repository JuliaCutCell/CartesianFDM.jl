module CartesianFDM

import Base: parent, size, getindex

using Reexport

@reexport using Symbolics
import Symbolics: Arr, scalarize

const subscripts = ("\u2081", "\u2082", "\u2083")
const TupleN{T,N} = NTuple{N,T}

export scalar, vector
export δ, σ
export BackwardArray, ForwardArray, MaskedArray
export δ⁻, δ⁺, σ⁻, σ⁺, μ
export gradient, divergence

include("symbolics.jl")
include("arrays.jl")
include("calculus.jl")

end
