abstract type BoundaryCondition end

struct Periodic <: BoundaryCondition end
struct Dirichlet <: BoundaryCondition end

periodic(args...) = Periodic()
dirichlet(args...) = Dirichlet()

