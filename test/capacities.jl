using Revise
using CartesianFDM

n = (6, 5)
top = (nonperiodic(), periodic())

ε = only(@variables ε)

# cell-centered volume capacity
mask = ((false, true), (false, true))
V = scalarfield(:V, n, top, ε; mask)

# face-centered volume capacities
W = vectorfield(:W, n)

# face-centered area capacities
mask = (((true, true), (false, true)),
        ((false, true), (true, true)))
A = vectorfield(:A, n, top; mask)

# cell-centered area capacities
tag = ntuple(i -> cell(n), length(n))
mask = (((false, true), (false, true)),
        ((false, true), (false, true)))
B = vectorfield(:B, n, top; mask, tag)

#
T = scalarfield(:T, n)
U = vectorfield(:U, n)

(; θ, δ, σ, ω) = operators(top, n)

