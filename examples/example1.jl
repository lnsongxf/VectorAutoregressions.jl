using VectorAutoregressions
using DataFrames

# Example from Lutkepohl (2005), Section 3.2.3, p.77
e1data = readtable("e1data.csv")
e1data[:date] = Date(e1data[:date])
e1data2 = e1data[Date(1960,12,1).<e1data[:date].< Date(1979,3,1), :]

datamat  = diff(log(Matrix(e1data2[:, 2:end])))
constant = true
trend    = false
varnames = string.(names(e1data2)[2:end])

lagselect = varselect(datamat, 8, constant, trend)
v = VAR(datamat, lagselect["AIC"], constant, trend, varnames)
stab_test = StabilityTest(v)
h = 4
corr_test = ResidualCorrelationTests(v, h)
