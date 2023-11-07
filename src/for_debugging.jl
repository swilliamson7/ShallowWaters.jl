using NetCDF, Parameters, Printf, Dates, Interpolations

using Enzyme#main
Enzyme.API.maxtypeoffset!(3500)

include("default_parameters.jl")
include("grid.jl")
include("constants.jl")
include("forcing.jl")
include("preallocate.jl")
include("model_setup.jl")
include("initial_conditions.jl")

include("time_integration.jl")
include("ghost_points.jl")
include("rhs.jl")
include("gradients.jl")
include("interpolations.jl")
include("PV_advection.jl")
include("continuity.jl")
include("bottom_drag.jl")
include("diffusion.jl")
include("tracer_advection.jl")

include("feedback.jl")
include("output.jl")
include("run_model.jl")
include("zanna_bolton_forcing.jl")
S = run_setup(nx=128);

using InteractiveUtils
@show @code_typed ZB_momentum(S.Prog.u, S.Prog.v, S, S.Diag)

@show @code_llvm ZB_momentum(S.Prog.u, S.Prog.v, S, S.Diag)