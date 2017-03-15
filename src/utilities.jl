function varstable(v::VAR; print = true)
    B = v.B
    p = v.lags
    constant = v.constant
    trend = v.trend
    startidx = constant + trend + 1
    A = B[startidx:end, :]
    Acomp = comp_matrix(A, p)
    eig_Acomp = eigs(Acomp)
    E = eig_Acomp[1]
    Emod = abs(E)

    if print
        @printf "==============================================\n"
        @printf "        Eigenvalues       Modulus \n"
        @printf "==============================================\n"
        for i = 1:length(E)
            if real(E[i]) > 0 && imag(E[i]) >= 0
                @printf "     %5.3f +   %5.3fi      %5.3f \n" real(E[i]) imag(E[i]) Emod[i]
            elseif real(E[i]) > 0 && imag(E[i]) < 0
                @printf "     %5.3f +   %5.3fi      %5.3f \n" real(E[i]) imag(E[i]) Emod[i]
            elseif real(E[i]) < 0 && imag(E[i]) >= 0
                @printf "    %5.3f +   %5.3fi      %5.3f \n" real(E[i]) imag(E[i]) Emod[i]
            else
                @printf "    %5.3f +  %5.3fi      %5.3f \n" real(E[i]) imag(E[i]) Emod[i]
            end
        end
        @printf "==============================================\n"

        test = (Emod .< 1)
        if all(test)
            @printf "All eigenvalues lie inside the unit circle. \n"
            @printf "VAR satisfies the stability condition. \n\n"
        else
            @printf "At least one eigenvalue lies outside the unit circle. \n"
            @printf "VAR does not satisfy the stability condition. \n\n"
        end
    end

    return (E, Emod)
end

"""
    loglik(T, K, SigmaU)

Calculate the log likelihood of [`VAR`](@ref) instance `v`.
"""
function loglik(T, K, SigmaU)
    # equivalent to Lutkepohl, p.89
    LL = -(T/2.0)*(logdet(SigmaU) + K*log(2*pi) + K)
    return LL
end

"""
    criteria(T, K, p, SigmaU)

Calculate lag selection criteria for [`VAR`](@ref) instance `v`.
"""
function criteria(T, K, p, SigmaU)
    # MLE estimate of SigmaU
    Sigma = ((T - K*p - 1)/T)*SigmaU
    # FPE
    FPE = det(Sigma)*(((T + K*p + 1)/(T - K*p - 1))^K)*1e11
    # AIC = log(det(Sigma_u)) + (2/T)*(# of parameters)
    AIC = log(det(Sigma)) + (2.0*p*K^2)/T
    # HQIC
    HQIC = log(det(Sigma)) + ((2.0*p*K^2)/T)*log(log(T))
    # SIC
    SIC = log(det(Sigma)) + ((p*K^2)/T)*log(T)

    return Dict("FPE" => FPE, "AIC" => AIC, "SIC" => SIC, "HQIC" => HQIC)
end

"""
    varselect(datamat, maxlag, constant, trend; print = false)

Calculate optimal lag length according to information criteria and FPE.
"""
function varselect(datamat, maxlag, constant, trend; print = false)
    critname = ["FPE", "AIC", "HQIC", "SIC"]
    crit = Array{Float64}(4, maxlag)
    for lag in 1:maxlag
        dmat = datamat[(maxlag+1-lag):end, :]
        nobs, K, T, Z, B, U, SigmaU, SigmaB, seB = var_ols(dmat, lag, constant=constant, trend=trend)
        infcrit = criteria(T, K, lag, SigmaU)
        crit[1, lag] = infcrit["FPE"]
        crit[2, lag] = infcrit["AIC"]
        crit[3, lag] = infcrit["HQIC"]
        crit[4, lag] = infcrit["SIC"]
    end
    lagselect = Dict(critname[i] => indmin(crit[i,:]) for i in 1:4)

    if print
        @printf "======================\n"
        @printf "  Criteria       Lag \n"
        @printf "======================\n"
        @printf "    FPE           %d\n" lagselect["FPE"]
        @printf "    AIC           %d\n" lagselect["AIC"]
        @printf "    HQIC          %d\n" lagselect["HQIC"]
        @printf "    SIC           %d\n" lagselect["SIC"]
        @printf "======================\n"
    end
    return lagselect
end
