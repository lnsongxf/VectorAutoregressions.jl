"""
    loglik(v::VAR)

Calculate the log likelihood of [`VAR`](@ref) instance `v`.
"""
function loglik(v::VAR)
    nobs, K = size(v.datamat)
    T = nobs - v.lags
    SigmaU = vcov(v)
    # equivalent to Lutkepohl, p.89
    LL = -(T/2.0)*(logdet(SigmaU) + K*log(2*pi) + K)
    return LL
end
