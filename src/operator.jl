# MonoDir
# Nonlocal
struct Univariate{D,F,B,O} <: Function
    dim::D
    forward::F
    backward::B
    fn::O
end

function (op::Univariate)(x::Field)
    (; dim, forward, backward, fn) = op
    (; data, tag) = x

    res = tag[dim] ?
        fn.((*).(forward, Ref(data))...) :
        fn.((*).(backward, Ref(data))...)

    Field(res, flip(tag, dim))
end

univariates(dims, eye, τ⁺, τ⁻, fn) =
    map(dims, τ⁺, τ⁻) do d, plus, minus
        Univariate(d, (eye, plus), (minus, eye), fn)
    end

# LocalOperator
# Can this be generalized to do upwinding?
struct Composite{S,O} <: Function
    sub::S
    fn::O
end

function (op::Composite)(xs::Field...)
    (; sub, fn) = op

    ys = broadcast(sub, xs) do s, x
        s(x)
    end

    tags = getproperty.(ys, Ref(:tag))
    @assert all(==(first(tags)), Base.tail(tags))

    Field(fn.(ys...), first(tags))
end

permanent((x⁻, x⁺), (y⁻, y⁺)) = x⁻ * y⁺ + y⁻ * x⁺

function operators(top, n)
    dims = ntuple(identity, length(top))
    eye = I(prod(n))
    τ⁺ = forwardshiftmatrices(top, n)
    τ⁻ = backwardshiftmatrices(top, n)

    θ = univariates(dims, eye, τ⁺, τ⁻, tuple)
    δ = univariates(dims, eye, τ⁺, τ⁻, -)
    σ = univariates(dims, eye, τ⁺, τ⁻, +)

    ω = map(θ) do op
        Composite((op, op), permanent)
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

