import Base: splat, Fix1
#
#(f::Fix1)(y...) = f.f(f.x, y...)
#
using CartesianFDM

n = (5, 5)
top = ntuple(nonperiodic, length(n))

#=
eye = ntuple(length(n)) do i
    Shift{0}(top[i], prod(n[1:i-1]), prod(n[i+1:end]))
end

θ⁻ = ntuple(length(n)) do i
    Shift{-1}(top[i], prod(n[1:i-1]), prod(n[i+1:end]))
end

θ⁺ = ntuple(length(n)) do i
    Shift{1}(top[i], prod(n[1:i-1]), prod(n[i+1:end]))
end

x = collect(only(@variables x[1:prod(n)]))

y⁺ = map(Ref(x) .|> θ⁺) do el
    reshape(el, n...)
end

y⁻ = map(Ref(x) .|> θ⁻) do el
    reshape(el, n...)
end

pluses = map(eachdim(n)) do i
    Shift{1}(top[i], prod(n[1:i-1]), prod(n[i+1:end]))
end

minuses = map(eachdim(n)) do i
    Shift{-1}(top[i], prod(n[1:i-1]), prod(n[i+1:end]))
end

#=
forward = ∘(splat(Base.Fix1(broadcast, tuple)), stack(identity, pluses), Ref)
backward = ∘(splat(Base.Fix1(broadcast, tuple)), stack(minuses, identity), Ref)
=#

combine(f, g...) = ∘(splat(Fix1(broadcast, f)), Stack(g...), Ref)

forwards = map(pluses) do plus
#    ∘(splat(Fix1(broadcast, SVector)), Stack(identity, plus), Ref)
    combine(SVector, identity, plus)
end

backwards = map(minuses) do minus
#    ∘(splat(Fix1(broadcast, SVector)), Stack(minus, identity), Ref)
    combine(SVector, minus, identity)
end

op = map(eachdim(n), forwards, backwards) do i, forward, backward
    StaggeredOperator(uniswitch(i), uniflip(i), forward, backward)
end

op[1](Φ)
=#

(; θ) = nonlocaloperators(top, n)

#=
Φ = scalarfield(:Φ, n)
U = vectorfield(:U, n)
=#

Ψ = TaggedVector(only(@variables Ψ[Base.OneTo.(n)...]) |> collect, (false, false))

θ[1](Ψ .- Ψ .^ 2)

splat(-).(θ[1](Ψ))

# syntactic sugar
δ = map(θ) do el
    ∘(Fix1(broadcast, splat(-)), el)
end

δ[1](Ψ)

