immutable StabilityTest
    stable::Bool
    eigenvals::Array{Number}
    eigenmod::Array{Float64}

    function StabilityTest(v::VAR)
        B = v.B
        p = v.p
        constant = v.constant
        trend = v.trend
        eigenvals, eigenmod = varstable(B, p, constant, trend)
        stable = all(eigenmod .< 1)
        new(stable, eigenvals, eigenmod)
    end
end

function varstable(B, p, constant, trend)
    startidx = constant + trend + 1
    A = B[startidx:end, :]
    Acomp = comp_matrix(A, p)
    eig_Acomp = eigs(Acomp)
    E = eig_Acomp[1]
    Emod = abs(E)
    return E, Emod
end

# if print
#     @printf "==============================================\n"
#     @printf "        Eigenvalues       Modulus \n"
#     @printf "==============================================\n"
#     for i = 1:length(E)
#         if real(E[i]) > 0 && imag(E[i]) >= 0
#             @printf "     %5.3f +   %5.3fi      %5.3f \n" real(E[i]) imag(E[i]) Emod[i]
#         elseif real(E[i]) > 0 && imag(E[i]) < 0
#             @printf "     %5.3f +   %5.3fi      %5.3f \n" real(E[i]) imag(E[i]) Emod[i]
#         elseif real(E[i]) < 0 && imag(E[i]) >= 0
#             @printf "    %5.3f +   %5.3fi      %5.3f \n" real(E[i]) imag(E[i]) Emod[i]
#         else
#             @printf "    %5.3f +  %5.3fi      %5.3f \n" real(E[i]) imag(E[i]) Emod[i]
#         end
#     end
#     @printf "==============================================\n"
#
#     test = (Emod .< 1)
#     if all(test)
#         @printf "All eigenvalues lie inside the unit circle. \n"
#         @printf "VAR satisfies the stability condition. \n\n"
#     else
#         @printf "At least one eigenvalue lies outside the unit circle. \n"
#         @printf "VAR does not satisfy the stability condition. \n\n"
#     end
# end


immutable ResidualCorrelationTests
    U::Matrix{Float64}
    Q::Float64
    pvalQ::Float64
    F::Float64
    pvalF::Float64

    function ResidualCorrelationTests(v::VAR, h::Int)
        Y = v.Y
        K = v.K
        p = v.p
        T = v.T
        U = v.U
        Q, dfQ, pvalQ = portmanteau_test(U, K, p, T, h)
        F, dfF1, dfF2, pvalF = bg_lm_test(Y, K, T, p, U, h)
        new(U, Q, pvalQ, F, pvalF)
    end
end

"""
    portmanteau_test(U, K, p, T, h)

Calculate the test statistic and p-value for a multivariate portmanteau test for residual autocorrelation.
Implements the adjusted test described on p.171 of Lutkepohl (2005).

"""
function portmanteau_test(U, K, p, T, h)
    C0 = (U'*U)/T
    invC0 = inv(C0)
    Q = 0.0
    for i in 1:h
        F = [zeros(i, T); eye(T-i) zeros(T-i, i)]
        Ci = (U'*F*U)/T
        Q += trace(Ci'*invC0*Ci*invC0)*(1.0/(T - i))
    end
    # This version of the test statistic is from Hosking (JASA 1980)
    Q *= T^2
    df = K^2*(h-p)
    pval = 1 - cdf(Chisq(df), Q)
    return Q, df, pval
end

"""
    bg_lm_test(Y, K, T, p, U, h)
"""
function bg_lm_test(Y, K, T, p, U, h)
    Y2 = Y[h+1:end, :]
    T2 = T - h
    Ylag = lag_matrix(Y2, p)
    Ulag = lag_matrix(U, h)
    RHS = [Ylag Ulag]
    LHS = U[h+1:end, :]
    BD = (RHS'*RHS)\(RHS'*LHS)
    E = U[h+1:end, :] - RHS*BD
    SigmaE = (E'*E)/T2
    SigmaU = (U'*U)/T

    s = sqrt((K^4*h^2 - 4)/(K^2 + (K*h)^2 - 5))
    N = T - K*p - 1 - K*h - (K - K*h + 1)/2.0
    df1 = K^2*h
    df2 = N*s - (K^2*h)/2.0 + 1
    F = ((det(SigmaU)/det(SigmaE))^(1/s) - 1)*(df2/df1)
    pval = 1 - cdf(FDist(df1, df2), F)
    return F, df1, df2, pval
end
