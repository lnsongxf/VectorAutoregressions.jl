using VectorAutoregressions
using DataFrames

# Example from Lutkepohl (2005), Section 3.2.3, p.77
e1data = readtable("e1data.csv")
e1data[:date] = Date(e1data[:date])
e1data2 = e1data[e1data[:date].< Date(1979,3,1), :]

datamat  = diff(log(Matrix(e1data2[:, 2:end])))
constant = true
trend    = false
varnames = string.(names(e1data2)[2:end])

lagselect = varselect(datamat, 8, constant, trend)
v = VAR(datamat, lagselect["AIC"], constant, trend, varnames)
stab_test = StabilityTest(v)
h = 4
corr_test = ResidualCorrelationTests(v, h)

P = chol(v.SigmaU)'
w = v.U*inv(P)
b1 = sum(w.^3, 1)/v.T
lambda_s = v.T*b1*b1'/6
b2 = sum(w.^4, 1)/v.T
lamda_k = v.T*(b2-3)*(b2-3)'/24
