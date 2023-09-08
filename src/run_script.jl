include("ShallowWaters.jl")
using .ShallowWaters 
using Enzyme#main

### gradient check, passed  

# S = ShallowWaters.run_check(nx=10, Ndays=1)
S = ShallowWaters.run_setup(nx=512, Ndays=365*10)
dS = deepcopy(S)

# dS.Prog.u[27, 27] = 1.0

ShallowWaters.autodiff(Reverse, ShallowWaters.time_integration_mine, Duplicated(S, dS))

# want_to_match = dS.Prog.u[27,27]
# save = S.Prog.u[27,27]

# steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# for_checking = []
# for s in steps 

#     S = ShallowWaters.run_enzyme(nx=50, Ndays=1)
#     S.Prog.u[27,27] = s
#     ShallowWaters.time_integration_debug(S)

#     temp = (S.Prog.u[27,27] - save) / s
#     push!(for_checking, temp)

# end 

