"""
Sarah addition -- structure to contain any variables that are parameters
but change (includes things related to optimzation/adjoints/checkpointing/etc.)
"""

@with_kw mutable struct VaryingParameters

    T=Float32                 # number format

    Tprog=T                   # number format for prognostic variables
    Tcomm=Tprog               # number format for ghost-point copies
    Tini=Tprog                # number format to reduce precision for initial conditions

    # NN FORCING OPTIONS
    weights_center::Array{Float32, 2} = zeros(Float32,1,17)
    weights_corner::Array{Float32, 2} = zeros(Float32,2,22)

    # PARAMETERS FOR ADJOINT METHOD
    data_steps::StepRange{Int,Int} = 0:1:0      # Timesteps where data exists
    data::Array{Float32} = zeros(1, 1, 1)       # model data
    J::Float64 = 0.                             # Placeholder for cost function evaluation
    j::Int = 1                                  # For keeping track of the entry in data
    average::Float64 = 0.0                      # Placeholder for computation of average values during checkpointing

    # CHECKPOINTING VARIABLES
    i::Int = 0                                  # Placeholder for current timestep, needed for Checkpointing.jl

end