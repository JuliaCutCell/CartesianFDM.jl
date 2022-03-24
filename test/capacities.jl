using Revise
using CartesianFDM

n = (6, 6)
top = (nonperiodic(), periodic())

V = scalarfield(:V, n, top, f = mask!)
A = vectorfield(:A, n, top, f = mask!)

P = scalarfield(:P, n)
ops = centeredoperators(n, top)

(; δ, σ) = ops

Q = σ[1] * P

#=
tag = ntuple(i -> cell(n)
=#
