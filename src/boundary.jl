abstract type BoundaryCondition end

###
struct Dirichlet <: BoundaryCondition end

const dir = Dirichlet()

###
struct Neumann <: BoundaryCondition end

const neu = Neumann()

###
struct Mixed{T} <: BoundaryCondition
    info::T
end

