using CartesianFDM

n = (6, 6)

T = scalar(:T, n)
D = scalar(:D, n)
V = scalar(:V, n)
A = map(enumerate(vector(:A, n))) do (d, X)
    μ(d, X)
end

# Dirichlet
gradD = gradient(A, V, T, D)
lapD = divergence(A, gradD)

# Neumann
gradN = gradient(A, V, 0, T, 0)
lapN = divergence(A, gradN, 0)

nothing

