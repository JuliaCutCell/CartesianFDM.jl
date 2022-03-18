struct FDMOperators{N,R,T,D,S,M}
    ρ::R
    τ::NTuple{2,NTuple{N,T}}
    δ::NTuple{2,NTuple{N,D}}
    σ::NTuple{2,NTuple{N,S}}
    μ::Tuple{M,NTuple{N,M}}
end

function fdmoperators(topo, n)
    ρ = spdiagm(0 => ones(Bool, prod(n)))
    τ = backwardshiftmatrices(topo, n), forwardshiftmatrices(topo, n)
    δ = Ref(ρ) .- τ[1], τ[2] .- Ref(ρ)
    σ = Ref(ρ) .+ τ[1], τ[2] .+ Ref(ρ)
    μ = maskmatrices(topo, n)
    FDMOperators(ρ, τ, δ, σ, μ)
end

"""

"""
_kron(opn) = opn

_kron(opn...) = kron(reverse(opn)...)

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
function forwardshiftmatrices(topo, n)
    opn = _forwardshift.(topo, n)
    eye = I.(n)
    _kron(opn, eye)
end

_forwardshift(::Topology, n::Int) =
    spdiagm(1 => ones(Bool, n-1))

_forwardshift(::Periodic, n::Int) =
    spdiagm(1-n => ones(Bool, 1),
            1 => ones(Bool, n-1))

"""

"""
function backwardshiftmatrices(topo, n)
    opn = _backwardshift.(topo, n)
    eye = I.(n)
    _kron(opn, eye)
end

_backwardshift(::Topology, n::Int) =
    spdiagm(-1 => ones(Bool, n-1))

_backwardshift(::Periodic, n::Int) =
    spdiagm(-1 => ones(Bool, n-1),
            n-1 => ones(Bool, 1))

"""

"""
function maskmatrices(topo, n)
    opn = _mask.(topo, n)
    μ⁰ = _kron(opn...)

    opn = _normal.(topo, n)
    eye = _tangent.(topo, n)
    μ¹ = _kron(opn, eye)

    μ⁰, μ¹
end

_mask(::Topology, n::Int) =
    spdiagm(0 => [i ≠ n for i in 1:n])

_mask(::Periodic, n::Int) =
    spdiagm(0 => ones(Bool, n))

_normal(::Topology, n::Int) =
    spdiagm(0 => [i ≠ 1 && i ≠ n for i in 1:n])

_normal(::Periodic, n::Int) =
    spdiagm(0 => ones(Bool, n))

_tangent(::Topology, n::Int) =
    spdiagm(0 => [i ≠ n for i in 1:n])

_tangent(::Periodic, n::Int) =
    spdiagm(0 => ones(Bool, n))

