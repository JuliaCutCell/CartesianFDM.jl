using LinearAlgebra
using CartesianFDM

# @variables ε f g h
# λ = @variables λ₋ λ₊

# 0 < f < h
# 0 < g < h
const ε = eps(Float64)
const f = 1 / 2
const g = 1 / 2
const h = 1.
const λ = [2., 3.]

const T = typeof(ε)

tocell = Base.Fix2(TaggedVector, (false,))
toface = Base.Fix2(TaggedVector, (true,))

# primary fields
V = tocell.([T[4h, 4h, 4h, 2h+2f, ε, ε, ε, ε, 2h+2g, 4h, 4h, 4h, ε],
             T[ε, ε, ε, 2h-2f, 4h, 4h, 4h, 4h, 2h-2g, ε, ε, ε, ε]])

A = toface.([Bool[0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
             Bool[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

B = tocell.([Bool[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
             Bool[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

W = toface.([T[2h, 4h, 4h, 3h+f, h+f, ε, ε, ε, h+g, 3h+g, 4h, 4h, 2h],
             T[ε, ε, ε, h-f, 3h-f, 4h, 4h, 4h, 3h-g, h-g, ε, ε, ε]])

# secondary fields (hard-coded for now)
Ω = tocell.([Bool[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
             Bool[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

Γ = tocell.([Bool[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
             Bool[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

Σ = toface.([Bool[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
             Bool[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])

# Boundary condition
DBC = Float64[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]

# Jump condition
ΔΨ = Float64[0, 0, 0, 3, 0, 0, 0, 0, -5, 0, 0, 0, 0]
ΔQ = tocell(Float64[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])

# Postprocessing
X = tocell.([T[2h, 6h, 10h, 13h+f, 18h, 22h, 26h, 30h, 35h-g, 38h, 42h, 46h, 50h],
             T[2h, 6h, 10h, 15h+f, 18h, 22h, 26h, 30h, 33h-g, 38h, 42h, 46h, 50h]])
Y = tocell.([T[0, 6h, 10h, 14h+2f, 18h, 22h, 26h, 30h, 34h-2g, 38h, 42h, 48h, 50h],
             T[2h, 6h, 10h, 14h+2f, 18h, 22h, 26h, 30h, 34h-2g, 38h, 42h, 46h, 50h]])

n = length(V[1])
top = ntuple(nonperiodic, 1)
(; θ) = nonlocaloperators(top, (n,))

arithmetic(x, y) = (x + y) / 2

δ(x) = Base.splat(∘(-, -)).(θ[1](x))
σ(x) = Base.splat(arithmetic).(θ[1](x))

Φ = tocell.([collect(only(@variables Φ₋[1:n])),
             collect(only(@variables Φ₊[1:n]))])

Ψ = tocell.([collect(only(@variables Ψ₋[1:n])),
             collect(only(@variables Ψ₊[1:n]))])

P = toface.([collect(only(@variables P₋[1:n])),
             collect(only(@variables P₊[1:n]))])

Q = toface.([collect(only(@variables Q₋[1:n])),
             collect(only(@variables Q₊[1:n]))])

grad = map(A, B, Φ, Ψ) do a, b, φ, ψ
    δ(b .* φ) + δ((σ(a) .- b) .* ψ) - σ(δ(a) .* ψ)
end

flux = map(λ, W, P) do lambda, w, p
    lambda .* p ./ w
end

div = map(B, Q) do b, q
    -b .* δ(q)
end

RH = map(A, B, Q) do a, b, q
    δ((σ(b) .- a) .* q) - σ(δ(b) .* q)
end

absdA = map(A) do a
    abs.(δ(a))
end

# active
function myfilter(f, a)
    r = eltype(a)[]
    for (b, el) in zip(f, a)
        b && push!(r, el)
    end
    r
end

Φ̂ = myfilter.(Ω, Φ)
Ψ̂ = myfilter.(Γ, Ψ)

P̂ = myfilter.(Σ, P)
ĝrad = myfilter.(Σ, grad)

Q̂ = myfilter.(Σ, Q)
f̂lux = myfilter.(Σ, flux)

#D̂ = myfilter.(Ω, D)
d̂iv = myfilter.(Ω, div)

# absdA[1] = absdA[2] where Γ[1] .&& Γ[2]
R̂H = myfilter(Γ[1] .&& Γ[2], RH[1] + RH[2] .- absdA[1] .* ΔQ)

# 𝒢
Ĝ = map(ĝrad, Φ̂) do p, phi
    Symbolics.jacobian(p, phi)
end

# ℋ
Ĥ = map(ĝrad, Ψ̂) do p, psi
    Symbolics.jacobian(p, psi)
end

Λ̂ = map(f̂lux, P̂) do q, p
    Symbolics.jacobian(q, p)
end

# -𝒟  (sanity check)
Ĝᵀ = map(d̂iv, Q̂) do d, q
    Symbolics.jacobian(d, q)
end

# 𝒥
#=
Ĵ = map(Q̂) do q
    Symbolics.jacobian(R̂H, q)
end
=#

#
Ψ̃ = myfilter.(Ref(Γ[1] .&& Γ[2]), Ψ)
ΔΨ̃ = myfilter(Γ[1] .&& Γ[2], ΔΨ)

Ψ̄ = [myfilter((Γ[1] .⊻ Γ[2]) .&& Γ[1], Ψ[1]),
     myfilter((Γ[1] .⊻ Γ[2]) .&& Γ[2], Ψ[2])]
D̄BC = [myfilter((Γ[1] .⊻ Γ[2]) .&& Γ[1], DBC),
       myfilter((Γ[1] .⊻ Γ[2]) .&& Γ[2], DBC)]

r̃hs = Ψ̃[2] .- Ψ̃[1] .- ΔΨ̃
r̄hs = vcat(Ψ̄[1] .- D̄BC[1], Ψ̄[2] .- D̄BC[2])

r̂hs = vcat(r̃hs, r̄hs)

ĉond = vcat(r̂hs, R̂H)

Ĵ = Symbolics.jacobian(ĉond, vcat(Q̂...))
N̂ = Symbolics.jacobian(ĉond, vcat(Ψ̂...))

b̂ = vcat(d̂iv[1], f̂lux[1] .- Q̂[1], ĝrad[1] .- P̂[1],
         d̂iv[2], f̂lux[2] .- Q̂[2], ĝrad[2] .- P̂[2],
         ĉond)
x̂ = vcat(Φ̂[1], P̂[1], Q̂[1], Φ̂[2], P̂[2], Q̂[2], Ψ̂[1], Ψ̂[2])

Â = Symbolics.jacobian(b̂, x̂)

#=
(_fun, _) = build_function(b̂, x̂, f, g, h, λ₋, λ₊)
(_jac, _) = build_function(Â, x̂, f, g, h, λ₋, λ₊)
=#

(_fun, _) = build_function(b̂, x̂)
(_jac, _) = build_function(Â, x̂)

fun = eval(_fun)
jac = eval(_jac)

#=
bʰ = fun(zeros(Float64, length(x̂)), 1 / 2, 1 / 2, 1., 1., 2.)
Aʰ = jac(zeros(Float64, length(x̂)), 1 / 2, 1 / 2, 1., 1., 2.)
=#

bʰ = fun(zeros(Float64, length(x̂)))
#=
Aʰ = jac(zeros(Float64, length(x̂)))
=#

Aʰ = map(Â) do el
    Float64(el.val)
end

xʰ = -Aʰ \ bʰ

# postprocessing
X̂ = myfilter.(Ω, X)
Ŷ = myfilter.(Γ, Y)

open("temp.txt", "w") do io
    for iter in zip.((Φ̂..., Ψ̂..., x̂), (X̂..., Ŷ..., xʰ))
        for (sym, num) in iter
            write(io, "$sym = $num\n")
        end
    end
end

dict = Dict{Num,Float64}()
for (num, val) in zip(x̂, xʰ)
    dict[num] = val
end

vec = [NTuple{2,Float64}[],
       NTuple{2,Float64}[],
       NTuple{2,Float64}[],
       NTuple{2,Float64}[]]

for (i, iter) in enumerate(zip.((Φ̂..., Ψ̂...), (X̂..., Ŷ...)))
    for (sym, num) in iter
        push!(vec[i], (num, dict[sym]))
    end
end

using Plots

fig = scatter()
for el in vec
    scatter!(fig, first.(el), last.(el))
end

fig
