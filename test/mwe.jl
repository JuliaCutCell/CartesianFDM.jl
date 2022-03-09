using CartesianFDM

n = (6, 6)
bc = (periodic(), dirichlet())

ops = fdmoperators(bc, n)

V = scalar(:V, n)
T = scalar(:T, n)
D = scalar(:D, n)

A = mask(ops, vector(:A, n))

G = gradient(ops, A, V, T, D)
L = divergence(ops, A, G)

