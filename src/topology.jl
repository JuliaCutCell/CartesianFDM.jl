abstract type Topology end

struct Periodic <: Topology end
struct NonPeriodic <: Topology end

periodic(args...) = Periodic()
nonperiodic(args...) = NonPeriodic()

isperiodic(::Topology) = false
isperiodic(::Periodic) = true

