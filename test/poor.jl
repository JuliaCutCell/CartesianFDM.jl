using CartesianFDM

tocell = Base.Fix2(TaggedVector, (false,))
toface = Base.Fix2(TaggedVector, (true,))

const MyType{N,T} = NTuple{N,Union{Nothing,T}}

levelset(x) = min(x-0.3125, 0.6875-x)
#levelset(x) = min(x-0.3, 0.7-x)

light = levelset
dark = ∘(-, levelset)

n = 13
top = nonperiodic()
(; θ) = nonlocaloperators((top,), (n,))
(θ,) = θ

x = coordinate(top, n)

# primary fields
function integrate(f, x::Vararg{T,2}) where {T}
    y = f.(x)

    all(iszero, y) && error("interface crosses both end points")

    maximum(y) < zero(T) && return x[2]-x[1], (x[1]+x[2])/2, nothing
    minimum(y) > zero(T) && return nothing, nothing, nothing

    xm = (abs(y[1])*x[2] + abs(y[2])*x[1]) / (abs(y[1]) + abs(y[2]))

    y[1] < zero(T) ? (xm-x[1], (x[1]+xm)/2, xm) : (x[2]-xm, (xm+x[2])/2, xm)
end

buf1 = [MyType{3,Float64}[integrate(light, x[i], x[i+1]) for i in 1:n-1],
        MyType{3,Float64}[integrate(dark, x[i], x[i+1]) for i in 1:n-1]]

## V-type volume capacities
V = map(buf1) do el
    tocell(vcat(first.(el), nothing))
end

integrate(::T, ::Vararg{NTuple{2,Nothing},2}) where {T} = nothing
integrate(::T, left::Tuple{T,Nothing}, right::Tuple{T,Nothing}) where {T} = right[1]-left[1]
integrate(::T, left::NTuple{2,Nothing}, right::NTuple{2,T}) where {T} = right[1]-right[2]
integrate(::T, left::NTuple{2,T}, right::NTuple{2,Nothing}) where {T} = left[2]-left[1]
integrate(x::T, left::Tuple{T,Nothing}, ::NTuple{2,Nothing}) where {T} = x-left[1]
integrate(x::T, ::NTuple{2,Nothing}, right::Tuple{T,Nothing}) where {T} = right[1]-x
integrate(::T, left::Tuple{T,Nothing}, right::NTuple{2,T}) where {T} = right[1]-left[1]
integrate(::T, left::NTuple{2,T}, right::Tuple{T,Nothing}) where {T} = right[1]-left[1]
#
#integrate(x::T, ::Nothing, ::Nothing) where {T} = nothing
#integrate(x::T, ::Nothing, y::T) where {T} = y-x
#integrate(x::T, y::T, ::Nothing) where {T} = x-y
#integrate(x::T, y::T, z::T) where {T} = z-y

second(iter) = iter[2]

buf2 = map(buf1) do el
#    vcat(nothing, second.(el), nothing)
    vcat((nothing, nothing), Base.tail.(el), (nothing, nothing))
end

# W-type volume capacities
W = map(buf2) do el
#    toface(Union{Nothing,Float64}[integrate(x[i], el[i], el[i+1]) for i in 1:n])
    toface(Union{Nothing,Float64}[integrate(x[i], el[i], el[i+1]) for i in 1:n])
end

# hard fix for now
#W[1][5] = W[1][9] = 0.03125
#W[2][4] = W[2][10] = 0.010416666666666666

heaviside(f, ::Nothing) = false

function heaviside(f, x)
    y = f(x)

    iszero(y) && error("interface crosses point")

    y < zero(y)
end

buf3 = map(buf1) do el
    vcat(second.(el), nothing)
end

## B-type volume capacities
B = tocell.([heaviside.(Ref(light), buf3[1]),
             heaviside.(Ref(dark), buf3[2])])

## A-type volume capacities
A = toface.([vcat(false, [heaviside(light, x[i]) for i in 2:n-1], false),
             vcat(false, [heaviside(dark, x[i]) for i in 2:n-1], false)])

## Postprocessing
X = map(buf1) do el
    vcat(second.(el), nothing)
end

Y = map(buf1) do el
    vcat(last.(el), nothing)
end

### hard-coded for now
Y[1][begin] = x[1]
Y[1][end-1] = x[end]

# indicator fields: where to...

## ... solve for Φ±
Ω = map(V) do el
    (!isnothing).(el)
end

## ... solve for Ψ±
Γ = map(A) do el
    vcat([el[i] ⊻ el[i+1] for i in 1:n-1], false)
end

## ... solve for P±/Q±
Σ = map(B) do el
    (!isnothing).(el)
end

## ... solve for Ψ₊ - Ψ₋
Θ = Γ[1] .&& Γ[2]

## ... solve for Dirichlet boundary condition
Ξ = [(Γ[1] .⊻ Γ[2]) .&& Γ[1],
     (Γ[1] .⊻ Γ[2]) .&& Γ[2]]

# Boundary conditions
DBC = vcat(1., zeros(n-3), 2., 0.)

# Jump conditions
ΔΨ = ones(n)
ΔQ = ones(n)

# symbolic variables
Φ = tocell.([collect(only(@variables Φ₋[1:n])),
             collect(only(@variables Φ₊[1:n]))])

Ψ = tocell.([collect(only(@variables Ψ₋[1:n])),
             collect(only(@variables Ψ₊[1:n]))])

P = toface.([collect(only(@variables P₋[1:n])),
             collect(only(@variables P₊[1:n]))])

Q = toface.([collect(only(@variables Q₋[1:n])),
             collect(only(@variables Q₊[1:n]))])

# operators
average(x::Vararg{T,N}) where {T,N} = sum(x) / N

δ(x) = Base.splat(∘(-, -)).(θ(x))
σ(x) = Base.splat(average).(θ(x))

# governing equations
grad = map(A, B, Φ, Ψ, P) do a, b, φ, ψ, p
    δ(b .* φ) + δ((σ(a) .- b) .* ψ) - σ(δ(a) .* ψ) .- p
end

λ = [2., 4.]

flux = map(λ, W, P, Q) do lambda, w, p, q
    map(w, p, q) do x, y, z
        isnothing(x) ? -z : lambda * y / x - z
    end
end

div = map(B, Q) do b, q
    -b .* δ(q)
end

boundary = map(A, B, Q) do a, b, q
    δ((σ(b) .- a) .* q) - σ(δ(b) .* q)
end

# select active variables
function mask(indicator, x)
    y = eltype(x)[]
    for (bool, el) in zip(indicator, x)
        bool && push!(y, el)
    end
    y
end

ΦΩ = mask.(Ω, Φ)
divΩ = mask.(Ω, div)

PΣ = mask.(Σ, P)
QΣ = mask.(Σ, Q)
gradΣ = mask.(Σ, grad)
fluxΣ = mask.(Σ, flux)

# all
ΨΓ = mask.(Γ, Ψ)

# boundary
ΨΞ = mask.(Ξ, Ψ)
DBCΞ = mask.(Ξ, Ref(DBC))

# interface
ΨΘ = mask.(Ref(Θ), Ψ)
RHΘ = mask(Θ, sum(boundary) .- abs.(δ(A[1])) .* ΔQ)
ΔΨΘ = mask(Θ, ΔΨ)

# symbolic assembly
bsym = vcat(divΩ[1], fluxΣ[1], gradΣ[1],
            divΩ[2], fluxΣ[2], gradΣ[2],
            ΨΘ[2] - ΨΘ[1] - ΔΨΘ,
            ΨΞ[1] - DBCΞ[1], ΨΞ[2] - DBCΞ[2],
            RHΘ)

xsym = vcat(ΦΩ[1], QΣ[1], PΣ[1],
            ΦΩ[2], QΣ[2], PΣ[2],
            ΨΓ[1], ΨΓ[2])

Asym = Symbolics.jacobian(bsym, xsym)

# numerical computation
(fun, _) = build_function(bsym, xsym)
fun = eval(fun)

bnum = fun(zeros(size(xsym)))

Anum = map(Asym) do el
    Float64(el.val)
end

xnum = -Anum \ bnum

# postprocessing
XΩ = Vector{Float64}.(mask.(Ω, X))
YΓ = Vector{Float64}.(mask.(Γ, Y))

dict = Dict{Num,Float64}()
for (sym, num) in zip(xsym, xnum)
    dict[sym] = num
end

vec = ntuple(i -> NTuple{2,Float64}[], 4)

for (v, iter) in zip(vec, zip.((ΦΩ..., ΨΓ...), (XΩ..., YΓ...)))
    for (sym, num) in iter
        push!(v, (num, dict[sym]))
    end
end

labels = ("Φ₋", "Φ₊", "Ψ₋", "Ψ₊")

using Plots

fig = scatter()

for (v, str) in zip(vec, labels)
    scatter!(fig, first.(v), last.(v), label=str)
end

fig

