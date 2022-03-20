import Symbolics: derivative

function linearize(::Star, top, n, ops, F, X)
    (τ⁻, τ⁺) = getproperty(ops, :τ)

    T = promote_type(typeof(F), typeof(X))
    diags = Dict{Int,T}()

    for i in eachindex(top)
        p = prod(n[1:i-1])
        ∂ = derivative.(F, τ⁻[i] * X)
        if isperiodic(top[i])
            D = map(CartesianIndices(n), reshape(∂, n...)) do ind, el
                ind[i] == 1 ? zero(el) : el
            end
            diags[-p] = D[begin+p:end]
            D = map(CartesianIndices(n), reshape(∂, n...)) do ind, el
                ind[i] != 1 ? zero(el) : el
            end
            q = (n[i]-1) * p
            diags[q] = D[begin:end-q]
        else
            diags[-p] = ∂[begin+p:end]
        end
    end

    diags[0] = derivative.(F, X)

    for i in eachindex(top)
        p = prod(n[1:i-1])
        ∂ = derivative.(F, τ⁺[i] * X)
        if isperiodic(top[i])
            D = map(CartesianIndices(n), reshape(∂, n...)) do ind, el
                ind[i] == n[i] ? zero(el) : el
            end
            diags[+p] = D[begin:end-p]
            D = map(CartesianIndices(n), reshape(∂, n...)) do ind, el
                ind[i] != n[i] ? zero(el) : el
            end
            q = (n[i]-1) * p
            diags[-q] = D[begin+q:end]
        else
            diags[+p] = ∂[begin:end-p]
        end
    end

#    spdiagm(diags...)
    diags
end
