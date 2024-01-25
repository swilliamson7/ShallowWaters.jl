# Assuming the information we have is solely a spacially averaged kinetic energy

include("../src/ShallowWaters.jl")
using .ShallowWaters
using Enzyme#main
using Checkpointing
using JLD2



energy_high_resolution = 
S = ShallowWaters.run_setup()