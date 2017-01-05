"""
Represents a Vector Autoregression (VAR) estimated using ordinary least squares (OLS).

Fields:
---
"""
immutable VAR
    datamat::Matrix{Float64}
    lags::Int
    constant::Bool
    trend::Bool
    varnames::Array{String, 1}
    Z::Matrix{Float64}
    B::Matrix{Float64}
    U::Matrix{Float64}
    SigmaU::Matrix{Float64}
    SigmaB::Matrix{Float64}
    seB::Matrix{Float64}

    function VAR(datamat::Matrix{Float64}, lags::Int, constant::Bool, trend::Bool, varnames::Array{String, 1}=[""])
        nobs, K, T, Z, B, U, SigmaU, SigmaB, seB = var_ols(datamat, lags, constant=constant, trend=trend)
        if varnames == [""]
            varnames = [string("y$i") for i in 1:K]
        end
        new(datamat, lags, constant, trend, varnames, Z, B, U, SigmaU, SigmaB, seB)
    end

    function VAR(datadf::DataFrame, lags::Int, constant::Bool, trend::Bool)
        datamat = Matrix(datadf)
        varnames = [string(i) for i in names(datadf)]
        VAR(datamat, lags, constant, trend, varnames)
    end

end # VAR type definition

varnames(v::VectorAutoregression.VAR) = v.varnames
coef(v::VectorAutoregression.VAR) = v.B
residuals(v::VectorAutoregression.VAR) = v.U
vcov(v::VectorAutoregression.VAR) = v.SigmaU
lags(v::VectorAutoregression.VAR) = v.lags



"""
    var_ols(datamat::Matrix, p::Int; constant::Bool=true, trend::Bool=false)

Estimate a VAR using OLS. Used as part of the inner constructor for the [`VAR`](@ref) tyoe.
"""
function var_ols(datamat::Matrix, p::Int; constant::Bool=true, trend::Bool=false)
    nobs, K = size(datamat)
    T = nobs - p
    Z = rhs_matrix(datamat, p, constant, trend)
    nparam = size(Z, 2)
    Y = datamat[p+1:end, :]
    B = (Z'*Z)\(Z'*Y) # each column corresponds to an equation
    U = Y - Z*B
    SigmaU = (U'*U)/(nobs - nparam)
    SigmaB = kron(SigmaU, inv(Z'*Z))  # following Lutkepohl(2005) p.80
    seB = sqrt(reshape(diag(SigmaB), nparam, K))
    return nobs, K, T, Z, B, U, SigmaU, SigmaB, seB
end
