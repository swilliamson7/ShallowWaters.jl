include("ShallowWaters.jl")
using .ShallowWaters

# this checks that my modified forward run still returns the same values as the 
# forward run originally written by MK

Prog = ShallowWaters.run_model(nx=50, Ndays=1)
S = ShallowWaters.run_check(nx=50, Ndays=1)