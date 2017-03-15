using VectorAutoregressions
using DataFrames

include("../src/varselect.jl")
# Example from Lutkepohl (2005), Section 3.2.3, p.77

e1data = readtable("e1data.csv")
e1data[:date] = Date(e1data[:date])
e1data2 = e1data[e1data[:date].< Date(1979,3,1), :]

datamat = diff(log(Matrix(e1data2[:, 2:end])))
lags = 2
constant = true
trend = false
varnames = string.(names(e1data2)[2:end])

lagselect = varselect(datamat, 8, constant, trend)
varout = VAR(datamat, lagselect["HQIC"], constant, trend, varnames)
E, Emod = varstable(varout);
