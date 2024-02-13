### Original ################################################################

# struct ModelSetup{T<:AbstractFloat,Tprog<:AbstractFloat}
#     parameters::Parameter
#     grid::Grid{T,Tprog}
#     constants::Constants{T,Tprog}
#     forcing::Forcing{T}
# end

# I think adding Prog and Diag structs to this will be enough to Checkpoint/use Enzyme

#############################################################################

# Structure that will contain additional terms needed for the adjoint experiments,
# such as a location to store cost function evaluation

# @with_kw mutable struct AdjointVariables
#     data_steps::Array{Int, 1} = [0]             # Timesteps where data exists
#     data::Array{Float32, 2} = [0. 0.;0. 0.]           # model data
#     J::Float64 = 0.                             # Placeholder for cost function evaluation
#     j::Int = 0                                  # For keeping track of the entry in data
#     i::Int = 0                                  # Placeholder for current timestep, needed for Checkpointing.jl
# end

# """
#     P = ProgVars{T}(u,v,η,sst)
# Struct containing the prognostic variables u,v,η and sst.
# """
mutable struct PrognosticVars{T<:AbstractFloat}
    u::Array{T,2}           # u-velocity
    v::Array{T,2}           # v-velocity
    η::Array{T,2}           # sea surface height / interface displacement
    sst::Array{T,2}         # tracer / sea surface temperature
end

mutable struct DiagnosticVars{T,Tprog}
    RungeKutta::RungeKuttaVars{Tprog}
    Tendencies::TendencyVars{Tprog}
    VolumeFluxes::VolumeFluxVars{T}
    Vorticity::VorticityVars{T}
    Bernoulli::BernoulliVars{T}
    Bottomdrag::BottomdragVars{T}
    ArakawaHsu::ArakawaHsuVars{T}
    Laplace::LaplaceVars{T}
    Smagorinsky::SmagorinskyVars{T}
    SemiLagrange::SemiLagrangeVars{T}
    PrognosticVarsRHS::PrognosticVars{T}        # low precision version
    ZBVars::ZBVars{T}                           # My addition
end

mutable struct ModelSetup{T<:AbstractFloat,Tprog<:AbstractFloat}
    parameters::Parameter
    grid::Grid{T,Tprog}
    constants::Constants{T,Tprog}
    forcing::Forcing{T}
    Prog::PrognosticVars{T}
    Diag::DiagnosticVars{T, Tprog}
    t::Int
    # Adjoint::AdjointVariables
end
