"""
    lag_matrix(X, p::Int)

Create a matrix of lagged values.

Arguments
---------
* `X` - the `Matrix` or `Vector` to be lagged (`X` may be a `DataFrame` as well).
* `p` - the number of lags

`X` should be of dimension T x K, where T is the number of observations (including presample values) and K is the number of endogenous variables.

Returns
-------
* A matrix of dimension (T-p) x (K*p)
"""
function lag_matrix{T<:Array}(X::T, p::Int)
    if isa(X, Vector)
        X = reshape(X, length(X), 1)
    end
    Traw, K = size(X)
    Xlag = zeros(Traw - p, K*p)
    for ii = 1:p
       Xlag[1:(Traw - p), (K*(ii - 1) + 1):(K*ii)] = X[(p+1-ii):(Traw-ii), 1:K]
    end
    return Xlag
end

function lag_matrix{T<:DataFrame}(X::T, p::Int)
    X2 = Matrix(X)
    return lag_matrix(X2, p)
end

"""
    rhs_matrix(X, p::Int, constant::Bool, trend::Bool)

Create matrix of RHS variables (notated Z in Lutkepohl (2005))
"""
function rhs_matrix(X::Matrix{Float64}, p::Int, constant::Bool, trend::Bool)
    Z = lag_matrix(X, p)
    if trend
        Z = hcat(collect(1.0:size(Z,1)), Z)
    end
    if constant
        Z = hcat(ones(Float64, size(Z, 1)), Z)
    end
    return Z
end

function rhs_matrix{T<:DataFrame}(X::T, p::Int, constant::Bool, trend::Bool)
    X2 = Matrix(X)
    return rhs_matrix(X2, p, constant, trend)
end

"""
    comp_matrix(A::Matrix, p)

Construct the companion matrix for `A`.
"""
function comp_matrix(A::Matrix, p)
    K = size(A, 2)
    I = [eye(K*(p-1)) zeros(K*(p-1), K)]
    Acomp = sparse(vcat(A', I))
    return Acomp
end
