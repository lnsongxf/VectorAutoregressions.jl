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

function show(io::IO, st::StabilityTest)
    E = st.eigenvals
    Emod = st.eigenmod
    println(io, " VAR Stability Check")
    println(io, "---------------------------------")
    println(io, "    Eigenvalues       Modulus ")
    println(io, "---------------------------------")
    for i = 1:length(E)
        println(io, lpad(string(round(E[i], 3)), 17), lpad(string(round(Emod[i], 3)), 11))
    end
    println(io, "-----------------------------------------")

    if st.stable
        println(io, "All eigenvalues lie inside the unit circle.")
        println(io, "VAR satisfies the stability condition.")
    else
        println(io, "At least one eigenvalue lies outside the unit circle.")
        println(io, "VAR does not satisfy the stability condition.")
    end
end



immutable ResidualCorrelationTests
    U::Matrix{Float64}
    h::Int
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
        new(U, h, Q, pvalQ, F, pvalF)
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

function show(io::IO, rct::ResidualCorrelationTests)
    println(io, " Residual Correlation Tests")
    println(io)
    println(io, " Multivariate portmanteau test")
    println(io, " (Null is no autocorrelation)")
    println(io, "-----------------------------------")
    println(io, "  Q statistic:", lpad(string(round(rct.Q, 3)), 12))
    println(io, "  P-value:    ", lpad(string(round(rct.pvalQ, 3)), 12))
    println(io)
    println(io)
    println(io, " Multivariate Breusch–Godfrey test")
    println(io, " (Null is no autocorrelation)")
    println(io, "-----------------------------------")
    println(io, "  F statistic:", lpad(string(round(rct.F, 3)), 12))
    println(io, "  P-value:    ", lpad(string(round(rct.pvalF, 3)), 12))
end
