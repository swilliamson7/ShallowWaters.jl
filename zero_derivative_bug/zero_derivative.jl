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

    # Check if adding Zanna Bolton forcing term 
    if S.parameters.zb_forcing_momentum
        ShallowWaters.ZB_momentum(u,v,S,Diag)
    end

    # adding the terms
    ShallowWaters.momentum_u!(Diag,S,t)
    ShallowWaters.momentum_v!(Diag,S,t)

end

# function continuity!(u::AbstractMatrix,
#     v::AbstractMatrix,
#     η::AbstractMatrix,
#     Diag::DiagnosticVars,
#     S::ModelSetup,
#     t::Int)

#     @unpack U,V,dUdx,dVdy = Diag.VolumeFluxes
#     @unpack nstep_advcor = S.grid
#     @unpack time_scheme,surface_relax,surface_forcing = S.parameters

#     # divergence of mass flux
#     ∂x!(dUdx,U)
#     ∂y!(dVdy,V)

#     if surface_relax
#     ShallowWaters.continuity_surf_relax!(η,Diag,S,t)
#     elseif surface_forcing
#     ShallowWaters.continuity_forcing!(Diag,S,t)
#     else
#     continuity_itself!(Diag,S,t)
#     end

# end

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
        ShallowWaters.ghost_points!(u,v,η,S)
        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        for rki = 1:RKo
            if rki > 1
                ShallowWaters.ghost_points!(u1,v1,η1,S)
            end

            # type conversion for mixed precision
            u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
            v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
            η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

            rhs_nonlinear!(u1rhs,v1rhs,η1rhs,Diag,S,t)          # momentum only
            ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)   # continuity equation

            if rki < RKo
                caxb!(u1,u,RKbΔt[rki],du)   #u1 .= u .+ RKb[rki]*Δt*du
                caxb!(v1,v,RKbΔt[rki],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
                caxb!(η1,η,RKbΔt[rki],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
            end

                axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη

        end

        t += dtint

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
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    seasonal_wind_x=false,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=nx,
    Ndays = Ndays,
    zb_forcing_dissipation=true,
    γ₀ = 0.3,
    data_steps=data_steps)

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
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    seasonal_wind_x=false,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=nx,
    Ndays = Ndays,
    zb_forcing_dissipation=true,
    γ₀ = 0.3,
    data_steps=data_steps)

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
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=nx,
        Ndays = Ndays,
        zb_forcing_dissipation=true,
        γ₀ = 0.3,
        data_steps=data_steps)

        S_inner.Prog.u[n, m] += s

        J_inner = checkpointed_integration(S_inner, revolve)

        push!(diffs, (J_inner - J_outer) / s)

    end

    return S, dS, enzyme_deriv, diffs

end

S, dS, enzyme_deriv, diffs = run_script()