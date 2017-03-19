isdefined(Base, :__precompile__) && __precompile__()

module VectorAutoregressions

using  DataFrames
using  Distributions
# using  Plots
import Base: show

export
#   Types
    VAR,
    StabilityTest,
    ResidualCorrelationTests,

#   Estimation functions
    var_ols,

#   Utilities
    lag_matrix,
    rhs_matrix,
    comp_matrix,
    varstable,
    loglik,
    criteria,
    varselect


include("var_ols.jl")
include("matrix_utilities.jl")
include("utilities.jl")
include("diagnostics.jl")

# include("var_bayes.jl")
# include("var_mf_ols.jl")
# include("var_mf_bayes.jl")
# include("irfs.jl")
# include("irf_bvar.jl")

end # module
