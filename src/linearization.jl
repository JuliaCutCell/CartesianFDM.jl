import Symbolics: derivative

function linearize(::Star, top, n, ops, F, X)
    (τ⁻, τ⁺) = getproperty(ops, :τ)

    T = promote_type(typeof(F), typeof(X))
    diags = Dict{Int,T}()

    for i in eachindex(top)
        p = prod(n[1:i-1])
        L = τ⁻[i] * X
        diags[-p] = derivative.(F, L)[begin+p:end]
    end

    diags[0] = derivative.(F, X)

    for i in eachindex(top)
        p = prod(n[1:i-1])
        U = τ⁺[i] * X
        diags[+p] = derivative.(F, U)[begin:end-p]
    end

    spdiagm(diags...)
end

