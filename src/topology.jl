abstract type Topology end

struct Periodic <: Topology end
struct NonPeriodic <: Topology end

periodic(args...) = Periodic()
nonperiodic(args...) = NonPeriodic()

isperiodic(::Topology) = false
isperiodic(::Periodic) = true

flip(tag::NTuple{N}, i) where {N} =
    ntuple(j -> j == i ? !tag[j] : tag[j], Val(N))

cell(::NTuple{N}) where {N} = ntuple(i -> false, N)

face(i, ::NTuple{N}) where {N} = ntuple(==(i), N)
face(n::NTuple{N}) where {N} = ntuple(Base.Fix2(face, n), N)

