"""
Represents a Vector Autoregression (VAR) estimated using ordinary least squares (OLS).

Fields:
---
"""
immutable VAR
    Y::Matrix{Float64}
    K::Int
    T::Int
    p::Int
    constant::Bool
    trend::Bool
    varnames::Vector{String}
    Z::Matrix{Float64}
    B::Matrix{Float64}
    A::Array{Float64, 3}
    U::Matrix{Float64}
    SigmaU::Matrix{Float64}
    seB::Matrix{Float64}
    LL::Float64

    function VAR(Y::Matrix{Float64}, p::Int, constant::Bool, trend::Bool, varnames::Vector{String}=[""])
        nobs, K, T, Z, B, A, U, SigmaU, SigmaB, seB = var_ols(Y, p, constant, trend)
        if varnames == [""]
            varnames = [string("Y$i") for i in 1:K]
        end
        LL = loglik(T, K, SigmaU)
        new(Y, K, T, p, constant, trend, varnames, Z, B, A, U, SigmaU, seB, LL)
    end
end # VAR type definition


"""
    var_ols(Y::Matrix, p::Int; constant::Bool=true, trend::Bool=false)

Estimate a VAR using OLS. Used as part of the inner constructor for the [`VAR`](@ref) tyoe.
"""
function var_ols(Y::Matrix{Float64}, p::Int, constant::Bool, trend::Bool)
    nobs, K = size(Y)
    T = nobs - p
    Z = rhs_matrix(Y, p, constant, trend)
    nparam = size(Z, 2)
    Y = Y[p+1:end, :]
    B = (Z'*Z)\(Z'*Y) # each column corresponds to an equation
    A = reshape(B[constant+trend+1:end, :]', K, K, p)
    U = Y - Z*B
    SigmaU = (U'*U)/(T - nparam)
    SigmaB = kron(SigmaU, inv(Z'*Z))  # following Lutkepohl(2005) p.80
    seB = sqrt(reshape(diag(SigmaB), nparam, K))
    return nobs, K, T, Z, B, A, U, SigmaU, SigmaB, seB
end
