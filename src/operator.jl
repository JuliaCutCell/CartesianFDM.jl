uniswitch(i) = Base.Fix2(getindex, i)

uniflip(i) = Base.Fix2(i) do x, j
    ntuple(length(x)) do k
        k == j ? !x[k] : x[k]
    end
end


"""

1. Non-local ;
1. Single argument (univariate) ;
1. Unidimensional (maybe not?).

!!! note

    Instead of `dim`, include a function to be applied to the tag.
    Useful to generalize to multiple dimension?

"""
struct StaggeredOperator{D,U,S,T,F} <: Function
    switch::D
    flip::U
    forward::S
    backward::T
    f::F
end

function (op::StaggeredOperator)(x::TaggedVector)
    (; switch, flip, forward, backward, f) = op
    (; data, tag) = x

    res = switch(tag) ?
        f.(broadcast.(forward, ntuple(_ -> Ref(data), length(forward)))...) :
        f.(broadcast.(backward, ntuple(_ -> Ref(data), length(backward)))...)

    TaggedVector(res, flip(tag))
end

#struct CenteredOperator{M,F} <: Function
#    maps::M
#    f::F
#end

struct CompositeOperator{U,F} <: Function
    unary::U
    f::F
end

function (op::CompositeOperator)(xs::TaggedVector...)
    (; unary, f) = op

    ys = broadcast.(unary, Ref.(xs))

    TaggedVector(f.(parent.(ys)...), tag(ys...))
end

tilde((x⁻, x⁺), (y⁻, y⁺)) = x⁻ * y⁺ + y⁻ * x⁺

unidim_bivar(dim, eye, plus, minus, f) =
    StaggeredOperator(uniswitch(dim),
                      uniflip(dim),
                      (eye, plus),
                      (minus, eye),
                      f)

function nonlocaloperators(top, n)
    dims = ntuple(identity, length(top))

    eye = LinearMap(I(prod(n)))
    τ⁺ = LinearMap.(forwardshiftmatrices(top, n))
    τ⁻ = LinearMap.(backwardshiftmatrices(top, n))

    # Staggered operators
    θ = unidim_bivar.(dims, Ref(eye), τ⁺, τ⁻, Ref(SVector))
    δ = unidim_bivar.(dims, Ref(eye), τ⁺, τ⁻, Ref(-))
    σ = unidim_bivar.(dims, Ref(eye), τ⁺, τ⁻, Ref(+))

    # Composite operators
    ω = map(θ) do op
        CompositeOperator((op, op), tilde)
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

