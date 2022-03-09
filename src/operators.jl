struct FDMOperators{N,R,T,D,S,M}
    ρ::R
    τ::NTuple{2,NTuple{N,T}}
    δ::NTuple{2,NTuple{N,D}}
    σ::NTuple{2,NTuple{N,S}}
    μ::NTuple{N,M}
end

function fdmoperators(bc, n)
    ρ = spdiagm(0 => ones(Bool, prod(n)))
    τ = backwardshiftmatrices(bc, n), forwardshiftmatrices(bc, n)
    δ = Ref(ρ) .- τ[1], τ[2] .- Ref(ρ)
    σ = Ref(ρ) .+ τ[1], τ[2] .+ Ref(ρ)
    μ = maskmatrices(bc, n)
    FDMOperators(ρ, τ, δ, σ, μ)
end

# ρ, (τ⁻, τ⁺), (δ⁻, δ⁺), (σ⁻, σ⁺), μ = unpack(op)
unpack(x::FDMOperators) = x.ρ, x.τ, x.δ, x.σ, x.μ

"""

"""
_kron(opn::NTuple{1}, eye::NTuple{1}) = opn

_kron(opn::NTuple{2}, eye::NTuple{2}) =
    kron(eye[2], opn[1]),
    kron(opn[2], eye[1])

_kron(opn::NTuple{3}, eye::NTuple{3}) =
    kron(eye[3], eye[2], opn[1]),
    kron(eye[3], opn[2], eye[1]),
    kron(opn[3], eye[2], eye[1])

"""

"""
function forwardshiftmatrices(bc, n)
    opn = _forwardshift.(bc, n)
    eye = I.(n)
    _kron(opn, eye)
end

_forwardshift(::BoundaryCondition, n::Int) =
    spdiagm(1 => ones(Bool, n-1))

_forwardshift(::Periodic, n::Int) =
    spdiagm(1-n => ones(Bool, 1),
            1 => ones(Bool, n-1))

"""

"""
function backwardshiftmatrices(bc, n)
    opn = _backwardshift.(bc, n)
    eye = I.(n)
    _kron(opn, eye)
end

_backwardshift(::BoundaryCondition, n::Int) =
    spdiagm(-1 => ones(Bool, n-1))

_backwardshift(::Periodic, n::Int) =
    spdiagm(-1 => ones(Bool, n-1),
            n-1 => ones(Bool, 1))

"""

"""
function maskmatrices(bc, n)
    opn = _normal.(bc, n)
    eye = _tangent.(bc, n)
    _kron(opn, eye)
end

_normal(::BoundaryCondition, n::Int) =
    spdiagm(0 => [i ≠ 1 && i ≠ n for i in 1:n])

_normal(::Periodic, n::Int) =
    spdiagm(0 => ones(Bool, n))

_tangent(::BoundaryCondition, n::Int) =
    spdiagm(0 => [i ≠ n for i in 1:n])

_tangent(::Periodic, n::Int) =
    spdiagm(0 => ones(Bool, n))
