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

"""
    criteria(v::VAR)

Calculate lag selection criteria for [`VAR`](@ref) instance `v`.
"""
function criteria(v::VAR)
    p, SigmaU = v.lags, v.SigmaU
    Traw, K = size(v.datamat)
    T = Traw - p
    # MLE estimate of SigmaU
    Sigma = ((T - K*p - 1)/T)*SigmaU

    # FPE
    FPE = det(Sigma)*(((T + K*p + 1)/(T - K*p - 1))^K)
    # AIC = log(det(Sigma_u)) + (2/T)*(# of parameters)
    AIC = log(det(Sigma)) + (2.0*p*K^2)/T
    # HQIC
    HQIC = log(det(Sigma)) + ((2.0*p*K^2)/T)*log(log(T))
    # SIC
    SIC = log(det(Sigma)) + ((p*K^2)/T)*log(T)

    return Dict("FPE" => FPE, "AIC" => AIC, "SIC" => SIC, "HQIC" => HQIC)
end
