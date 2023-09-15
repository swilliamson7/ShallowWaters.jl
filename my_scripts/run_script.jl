include("../src/ShallowWaters.jl")
using .ShallowWaters 
using Enzyme#main

runlist = filter(x->startswith(x,"run"),readdir(pwd()))
existing_runs = [parse(Int,id[4:end]) for id in runlist] 
# P = ShallowWaters.run_model(nx=128, Ndays=365)

# Checking my adjusted function output versus MK output,
# need to change ModelSetup definition before running 
# mk_run_model
# passed 
# P, energy = ShallowWaters.run_model(nx=50, Ndays=1)

### gradient check, passed  
# S = ShallowWaters.run_setup(nx=50, Ndays=1)
# dS = deepcopy(S)

# dS.Prog.u[27, 27] = 1.0

# ShallowWaters.autodiff(Reverse, ShallowWaters.time_integration_nofeedback, Duplicated(S, dS))

# want_to_match = dS.Prog.u[27,27]
# save = S.Prog.u[27,27]

# steps = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# for_checking = []
# for s in steps 

#     S = ShallowWaters.run_setup(nx=50, Ndays=1)
#     S.Prog.u[27,27] = s
#     ShallowWaters.time_integration_nofeedback(S)

#     temp = (S.Prog.u[27,27] - save) / s
#     push!(for_checking, temp)

# end 

