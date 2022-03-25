"""

1. Non-local ;
1. Single argument (univariate) ;
1. Unidimensional.

"""
struct NonLocalOperator{D,S,T,F} <: Function
    dim::D
    forward::S
    backward::T
    f::F
end

function (op::NonLocalOperator)(x::TaggedVector)
    (; dim, forward, backward, f) = op
    (; data, tag) = x

    res = tag[dim] ?
        f.(broadcast.(forward, ntuple(_ -> Ref(data), length(forward)))...) :
        f.(broadcast.(backward, ntuple(_ -> Ref(data), length(backward)))...)

    TaggedVector(res, flip(tag, dim))
end

struct CompositeOperator{U,F} <: Function
    unary::U
    f::F
end

function (op::CompositeOperator)(xs::TaggedVector...)
    (; unary, f) = op

    ys = broadcast.(unary, Ref.(xs))

    TaggedVector(f.(parent.(ys)...), tag(ys...))
end

permanent((x⁻, x⁺), (y⁻, y⁺)) = x⁻ * y⁺ + y⁻ * x⁺

_tobinary(dim, eye, plus, minus, f) =
    NonLocalOperator(dim, (eye, plus), (minus, eye), f)

function nonlocaloperators(top, n)
    dims = ntuple(identity, length(top))
    eye = LinearMap(I(prod(n)))
    τ⁺ = LinearMap.(forwardshiftmatrices(top, n))
    τ⁻ = LinearMap.(backwardshiftmatrices(top, n))

    θ = _tobinary.(dims, Ref(eye), τ⁺, τ⁻, Ref(tuple))
    δ = _tobinary.(dims, Ref(eye), τ⁺, τ⁻, Ref(-))
    σ = _tobinary.(dims, Ref(eye), τ⁺, τ⁻, Ref(+))

    ω = map(θ) do op
        CompositeOperator((op, op), permanent)
    end

    (; θ, δ, σ, ω)
end

#UnidimOperator
#=
function size(mat::Operator)
    (; dir, forward, backward) = mat
    @assert size(forward) == size(backward)
    size(backward)
end

function (*)(mat::Operator, vec::Field)
    (; dir, forward, backward) = mat
    (; data, tag) = vec
    Field(tag[dir] ? forward * data : backward * data,
          flip(tag, dir))
end
=#

#=
struct CenteredOperators{N,D,S}
    δ::NTuple{N,D}
    σ::NTuple{N,S}
end

function centeredoperators(n, top)
    ρ = spdiagm(0 => ones(Bool, prod(n)))
    τ = forwardshiftmatrices(top, n), backwardshiftmatrices(top, n)

    δ = τ[1] .- Ref(ρ), Ref(ρ) .- τ[2]
    σ = τ[1] .+ Ref(ρ), Ref(ρ) .+ τ[2]

    dims = ntuple(identity, length(n))

    ops = map((δ, σ)) do (forward, backward)
        map(dims, forward, backward) do dir, plus, minus
            Operator(dir, plus, minus)
        end
    end

    CenteredOperators(ops...)
end
=#

