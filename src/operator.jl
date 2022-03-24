struct Operator{T,F<:AbstractMatrix{T},B<:AbstractMatrix{T}} <: AbstractMatrix{T}
    dir::Int
    forward::F
    backward::B
end

direction(mat::Operator) = mat.dir

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

