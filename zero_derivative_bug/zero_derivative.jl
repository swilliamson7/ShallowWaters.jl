# Notes
# 1. The entry S.parameters.data needs to contain only the data for timesteps
# where we want it. This can be accomplished by writing S.parameters.data = data[steps where we want it]
# 2.If there's an error with advection_coriolis after trying to run autodiff, this means
# the data_steps doesn't contain a timestep that actually happens during the integration, or at
# that seems to be what's happening

include("../src/ShallowWaters.jl")
using .ShallowWaters

using Enzyme#main
using Checkpointing

using InteractiveUtils

Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.runtimeActivity!(true)

using Parameters

@with_kw mutable struct Parameter

    T=Float32                 # number format

    Tprog=T                   # number format for prognostic variables
    Tcomm=Tprog               # number format for ghost-point copies
    Tini=Tprog                # number format to reduce precision for initial conditions

    # DOMAIN RESOLUTION AND RATIO
    nx::Int=128                         # number of grid cells in x-direction
    Lx::Int=3840e3                      # length of the domain in x-direction [m]
    L_ratio::Int=1                      # Domain aspect ratio of Lx/Ly

    # PHYSICAL CONSTANTS
    g::Float32=9.81                       # gravitational acceleration [m^2/s] 
    H::Float32=500.                        # layer thickness at rest [m]
    ρ::Float32=1e3                         # water density [kg/m^3]
    ϕ::Float32=35.                         # central latitude of the domain (for coriolis) [°]
    ω::Float32=2π/(24*3600)                # Earth's angular frequency [s^-1]
    R::Float32=6.371e6                     # Earth's radius [m]

    # SCALE
    scale::Float32=2^6                     # multiplicative scale for the momentum equations u,v
    scale_sst::Float32=2^15                # multiplicative scale for sst

    # BOTTOM TOPOGRAPHY OPTIONS
    topography::String="flat"         # "ridge", "seamount", "flat", "ridges", "bathtub"

    # TIME STEPPING OPTIONS
    time_scheme::String="RK"            # Runge-Kutta ("RK") or strong-stability preserving RK
                                        # "SSPRK2","SSPRK3","4SSPRK3"
    RKo::Int=4                          # Order of the RK time stepping scheme (2, 3 or 4)
    cfl::Float32=0.9                       # CFL number (1.0 recommended for RK4, 0.6 for RK3)
    Ndays::Float32=5                       # number of days to integrate for
    nstep_diff::Int=1                   # diffusive part every nstep_diff time steps.
    nstep_advcor::Int=0                 # advection and coriolis update every nstep_advcor time steps.
                                        # 0 means it is included in every RK4 substep

    # BOUNDARY CONDITION OPTIONS
    bc::String="nonperiodic"            # "periodic" or anything else for nonperiodic
    α::Float32=0.                          # lateral boundary condition parameter
                                        # 0 free-slip, 0<α<2 partial-slip, 2 no-slip

    # MOMENTUM ADVECTION OPTIONS
    adv_scheme::String="Sadourny"       # "Sadourny" or "ArakawaHsu"
    dynamics::String="nonlinear"        # "linear" or "nonlinear"

    # DIFFUSION OPTIONS
    diffusion::String="constant"        # "Smagorinsky" or "constant", biharmonic in both cases
    νB::Float32=500.0                      # [m^2/s] scaling constant for constant biharmonic diffusion

end

function model_setup(::Type{T}=Float32;     # number format
    kwargs...                               # all additional parameters
    ) where {T<:AbstractFloat}

    P = Parameter(T=T;kwargs...)
    return model_setup(T,P)
end

function model_setup(P::Parameter)
    @unpack T = P
    return model_setup(T,P)
end

function model_setup(::Type{T},P::Parameter) where {T<:AbstractFloat}

    @unpack Tprog = P

    G = ShallowWaters.Grid{T,Tprog}(P)
    C = ShallowWaters.Constants{T,Tprog}(P,G)
    F = ShallowWaters.Forcing{T}(P,G)

    Prog = ShallowWaters.initial_conditions(Tprog,G,P,C)
    Diag = ShallowWaters.preallocate(T,Tprog,G)

    S = ModelSetup{T,Tprog}(P,G,C,F,Prog,Diag,0)

    return S

end

function axb!(a::Matrix{T},x::Real,b::Matrix{T}) where {T<:AbstractFloat}
    m,n = size(a)
    @boundscheck (m,n) == size(b) || throw(BoundsError())

    xT = convert(T,x)

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
           a[i,j] += xT*b[i,j]
        end
    end
end

function caxb!(c::Array{T,2},a::Array{T,2},x::T,b::Array{T,2}) where {T<:AbstractFloat}
    m,n = size(a)
    @boundscheck (m,n) == size(b) || throw(BoundsError())
    @boundscheck (m,n) == size(c) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
           c[i,j] = a[i,j] + x*b[i,j]
        end
    end
end

function rhs_nonlinear!(u::AbstractMatrix,
    v::AbstractMatrix,
    η::AbstractMatrix,
    Diag,
    S,
    t::Int)

    @unpack h,h_u,h_v,U,V = Diag.VolumeFluxes
    @unpack H = S.forcing
    @unpack ep = S.grid

    ShallowWaters.UVfluxes!(u,v,η,Diag,S)              # U,V needed for PV advection and in the continuity equation
    if S.grid.nstep_advcor == 0              # evaluate every RK substep
        ShallowWaters.advection_coriolis!(u,v,η,Diag,S)    # PV and non-linear Bernoulli terms
    end
    ShallowWaters.PVadvection!(Diag,S)                 # advect the PV with U,V

    # Bernoulli potential - recalculate for new η, KEu,KEv are only updated in advection_coriolis
    @unpack p,KEu,KEv,dpdx,dpdy = Diag.Bernoulli
    @unpack g,scale,scale_inv = S.constants
    g_scale = g*scale
    ShallowWaters.bernoulli!(p,KEu,KEv,η,g_scale,ep,scale_inv)
    ShallowWaters.∂x!(dpdx,p)
    ShallowWaters.∂y!(dpdy,p)

    # adding the terms
    ShallowWaters.momentum_u!(Diag,S,t)
    ShallowWaters.momentum_v!(Diag,S,t)

end

function continuity!(u::AbstractMatrix,
    v::AbstractMatrix,
    η::AbstractMatrix,
    Diag,
    S,
    t::Int)

    @unpack U,V,dUdx,dVdy = Diag.VolumeFluxes
    @unpack nstep_advcor = S.grid
    @unpack time_scheme,surface_relax,surface_forcing = S.parameters

    # divergence of mass flux
    ShallowWaters.∂x!(dUdx,U)
    ShallowWaters.∂y!(dVdy,V)

    ShallowWaters.continuity_itself!(Diag,S,t)

end

function checkpointed_integration(S, scheme)

    # setup
    Diag = S.Diag
    Prog = S.Prog

    @unpack u,v,η,sst = Prog
    @unpack u0,v0,η0 = Diag.RungeKutta
    @unpack u1,v1,η1 = Diag.RungeKutta
    @unpack du,dv,dη = Diag.Tendencies
    @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
    @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

    @unpack um,vm = Diag.SemiLagrange

    @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
    @unpack time_scheme,compensated = S.parameters
    @unpack RKaΔt,RKbΔt = S.constants
    @unpack Δt_Δ,Δt_Δs = S.constants

    @unpack nt,dtint = S.grid
    @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(Diag.VolumeFluxes.h,η,S.forcing.H)
    ShallowWaters.Ix!(Diag.VolumeFluxes.h_u,Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(Diag.VolumeFluxes.h_v,Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(Diag.Vorticity.h_q,Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)

    ShallowWaters.advection_coriolis!(urhs,vrhs,ηrhs,Diag,S)
    ShallowWaters.PVadvection!(Diag,S)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)

    # run integration loop with checkpointing
    loop(S, scheme)

    return S.parameters.J

end

function loop(S,scheme)
    
    for S.parameters.i = 1:S.grid.nt

        Diag = S.Diag
        Prog = S.Prog
    
        @unpack u,v,η,sst = Prog
        @unpack u0,v0,η0 = Diag.RungeKutta
        @unpack u1,v1,η1 = Diag.RungeKutta
        @unpack du,dv,dη = Diag.Tendencies
        @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
        @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies
    
        @unpack um,vm = Diag.SemiLagrange
    
        @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
        @unpack time_scheme,compensated = S.parameters
        @unpack RKaΔt,RKbΔt = S.constants
        @unpack Δt_Δ,Δt_Δs = S.constants
    
        @unpack nt,dtint = S.grid
        @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid
        t = S.t
        i = S.parameters.i

        # ghost point copy for boundary conditions
        # ShallowWaters.ghost_points!(u,v,η,S)
        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        for rki = 1:RKo
            # if rki > 1
            #     ShallowWaters.ghost_points!(u1,v1,η1,S)
            # end

            # type conversion for mixed precision
            u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
            v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
            η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

            rhs_nonlinear!(u1rhs,v1rhs,η1rhs,Diag,S,t)          # momentum only
            
            @unpack U,V,dUdx,dVdy = Diag.VolumeFluxes
            @unpack nstep_advcor = S.grid
            @unpack time_scheme,surface_relax,surface_forcing = S.parameters
        
            # divergence of mass flux
            # ShallowWaters.∂x!(dUdx,U)
            m,n = size(dUdx)
            @boundscheck (m+1,n) == size(U) || throw(BoundsError())
            @inbounds for j ∈ 1:n, i ∈ 1:m
                dUdx[i,j] = U[i+1,j] - U[i,j]
            end

            # ShallowWaters.∂y!(dVdy,V)
            m,n = size(dVdy)
            @boundscheck (m,n+1) == size(V) || throw(BoundsError())
            @inbounds for j ∈ 1:n, i ∈ 1:m
                dVdy[i,j] = V[i+1,j] - V[i,j]
            end
        
            ShallowWaters.continuity_itself!(Diag,S,t)

            #if rki < RKo
               caxb!(u1,u,RKbΔt[2],du)   #u1 .= u .+ RKb[rki]*Δt*du
               caxb!(v1,v,RKbΔt[2],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
               caxb!(η1,η,RKbΔt[2],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
            #end

                axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη

        end

        # t += dtint

        # Cost function evaluation #####################################################################

        temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S.Prog.u,
        S.Prog.v,
        S.Prog.η,
        S.Prog.sst,S)...)

        S.parameters.J += sum(temp.η)

        ################################################################################################

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

    end

    return nothing

end

function run_script()

    # 225 steps = 1 day of integration in the 128 model

    Ndays=5
    data_steps = 1:1:88
    nx=10

    S = ShallowWaters.model_setup(output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    Lx=3840e3,
    seasonal_wind_x=false,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=nx,
    Ndays = Ndays)

    dS = Enzyme.Compiler.make_zero(Core.Typeof(S), IdDict(), S)
    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false
    )

    autodiff(Enzyme.ReverseWithPrimal, checkpointed_integration, Active, Duplicated(S, dS), Const(revolve))

    ###### The remainder runs a finite difference check ###########################

    n = 5
    m = 5
    enzyme_deriv = dS.Prog.u[n, m]

    steps = [50, 40, 30, 20, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    S_outer = ShallowWaters.model_setup(output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    Lx=3840e3,
    seasonal_wind_x=false,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=nx,
    Ndays = Ndays)

    snaps = Int(floor(sqrt(S_outer.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S_outer.grid.nt, snaps; 
        verbose=1, 
        gc=true, 
        write_checkpoints=false
    )

    J_outer = checkpointed_integration(S_outer, revolve)

    diffs = []

    for s in steps

        S_inner = ShallowWaters.model_setup(output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        Lx=3840e3,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=nx,
        Ndays = Ndays)

        S_inner.Prog.u[n, m] += s

        J_inner = checkpointed_integration(S_inner, revolve)

        push!(diffs, (J_inner - J_outer) / s)

    end

    return S, dS, enzyme_deriv, diffs

end

S, dS, enzyme_deriv, diffs = run_script()