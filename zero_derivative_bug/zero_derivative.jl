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

function model_setup(::Type{T}=Float32;     # number format
    kwargs...                               # all additional parameters
    ) where {T<:AbstractFloat}

    P = ShallowWaters.Parameter(T=T;kwargs...)
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

    # Check if adding Zanna Bolton forcing term 
    if S.parameters.zb_forcing_momentum
        ShallowWaters.ZB_momentum(u,v,S,Diag)
    end

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

    if surface_relax
    ShallowWaters.continuity_surf_relax!(η,Diag,S,t)
    elseif surface_forcing
    ShallowWaters.continuity_forcing!(Diag,S,t)
    else
    ShallowWaters.continuity_itself!(Diag,S,t)
    end

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
            continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)             # continuity equation

            #if rki < RKo
               caxb!(u1,u,RKbΔt[2],du)   #u1 .= u .+ RKb[rki]*Δt*du
               caxb!(v1,v,RKbΔt[2],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
               caxb!(η1,η,RKbΔt[2],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
            #end

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

function setup() 
    Main.ShallowWaters.ModelSetup{Float32, Float32}(Main.ShallowWaters.Parameter
    T: Float32 <: AbstractFloat
    Tprog: Float32 <: AbstractFloat
    Tcomm: Float32 <: AbstractFloat
    Tini: Float32 <: AbstractFloat
    nx: Int64 10
    Lx: Int64 3840000
    L_ratio: Int64 1
    g: Float32 9.81f0
    H: Float32 500.0f0
    ρ: Float32 1000.0f0
    ϕ: Float32 35.0f0
    ω: Float32 7.2722054f-5
    R: Float32 6.371f6
    scale: Float32 64.0f0
    scale_sst: Float32 32768.0f0
    wind_forcing_x: String "double_gyre"
    wind_forcing_y: String "constant"
    Fx0: Float32 0.12f0
    Fy0: Float32 0.0f0
    seasonal_wind_x: Bool false
    seasonal_wind_y: Bool false
    ωFx: Float32 2.0f0
    ωFy: Float32 2.0f0
    topography: String "flat"
    topo_ridges_positions: Array{Float64}((4,)) [0.05, 0.25, 0.45, 0.9]
    topo_height: Float32 100.0f0
    topo_width: Float32 300000.0f0
    surface_relax: Bool false
    t_relax: Float32 100.0f0
    η_refh: Float32 5.0f0
    η_refw: Float32 50000.0f0
    surface_forcing: Bool false
    ωFη: Float32 1.0f0
    A: Float32 3.0f-5
    ϕk: Float32 35.0f0
    wk: Float32 10000.0f0
    time_scheme: String "RK"
    RKo: Int64 4
    RKs: Int64 3
    RKn: Int64 5
    cfl: Float32 0.9f0
    Ndays: Float32 5.0f0
    nstep_diff: Int64 1
    nstep_advcor: Int64 0
    compensated: Bool false
    bc: String "nonperiodic"
    α: Float32 2.0f0
    zb_forcing_momentum: Bool false
    zb_forcing_dissipation: Bool true
    zb_filtered: Bool true
    N: Int64 1
    γ₀: Float32 0.3f0
    data_steps: StepRange{Int64, Int64}
    data: Array{Float64}((1, 1, 1)) [0.0;;;]
    J: Float64 -1.0753283277153969e-5
    j: Int64 1
    i: Int64 88
    adv_scheme: String "Sadourny"
    dynamics: String "nonlinear"
    bottom_drag: String "quadratic"
    cD: Float32 1.0f-5
    τD: Float32 300.0f0
    diffusion: String "constant"
    νB: Float32 500.0f0
    cSmag: Float32 0.15f0
    tracer_advection: Bool true
    tracer_relaxation: Bool true
    tracer_consumption: Bool false
    sst_initial: String "waves"
    sst_rect_coords: Array{Float64}((4,)) [0.0, 0.15, 0.0, 1.0]
    Uadv: Float32 0.2f0
    SSTmax: Float32 1.0f0
    SSTmin: Float32 -1.0f0
    τSST: Float32 100.0f0
    jSST: Float32 365.0f0
    SSTw: Float32 500000.0f0
    SSTϕ: Float32 0.5f0
    SSTwaves_ny: Float32 4.0f0
    SSTwaves_nx: Float32 4.0f0
    SSTwaves_p: Float32 0.5f0
    output: Bool false
    output_vars: Array{String}((4,))
    output_dt: Float32 24.0f0
    outpath: String "/Users/swilliamson/Documents/GitHub/ShallowWaters_work/ShallowWaters.jl/zero_derivative_bug"
    compression_level: Int64 3
    return_time: Bool false
    initial_cond: String "rest"
    initpath: String "/Users/swilliamson/Documents/GitHub/ShallowWaters_work/ShallowWaters.jl/zero_derivative_bug"
    init_run_id: Int64 0
    init_starti: Int64 -1
    get_id_mode: String "continue"
    run_id: Int64 -1
    init_interpolation: Bool true
  , Main.ShallowWaters.Grid{Float32, Float32}
    nx: Int64 10
    Lx: Int64 3840000
    L_ratio: Int64 1
    bc: String "nonperiodic"
    g: Float32 9.81f0
    H: Float32 500.0f0
    cfl: Float32 0.9f0
    Ndays: Float32 5.0f0
    nstep_diff: Int64 1
    nstep_advcor: Int64 0
    Uadv: Float32 0.2f0
    output_dt: Float32 24.0f0
    ω: Float32 7.2722054f-5
    ϕ: Float32 35.0f0
    R: Float32 6.371f6
    scale: Int64 64
    Δ: Float32 384000.0f0
    ny: Int64 10
    Ly: Float64 3.84e6
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    nT: Int64 100
    nu: Int64 90
    nv: Int64 90
    nq: Int64 121
    x_T: Array{Float64}((10,)) [192000.0, 576000.0, 960000.0, 1.344e6, 1.728e6, 2.112e6, 2.496e6, 2.88e6, 3.264e6, 3.648e6]
    y_T: Array{Float64}((10,)) [192000.0, 576000.0, 960000.0, 1.344e6, 1.728e6, 2.112e6, 2.496e6, 2.88e6, 3.264e6, 3.648e6]
    x_u: Array{Float64}((9,)) [384000.0, 768000.0, 1.152e6, 1.536e6, 1.92e6, 2.304e6, 2.688e6, 3.072e6, 3.456e6]
    y_u: Array{Float64}((10,)) [192000.0, 576000.0, 960000.0, 1.344e6, 1.728e6, 2.112e6, 2.496e6, 2.88e6, 3.264e6, 3.648e6]
    x_v: Array{Float64}((10,)) [192000.0, 576000.0, 960000.0, 1.344e6, 1.728e6, 2.112e6, 2.496e6, 2.88e6, 3.264e6, 3.648e6]
    y_v: Array{Float64}((9,)) [384000.0, 768000.0, 1.152e6, 1.536e6, 1.92e6, 2.304e6, 2.688e6, 3.072e6, 3.456e6]
    x_q: Array{Float64}((11,)) [0.0, 384000.0, 768000.0, 1.152e6, 1.536e6, 1.92e6, 2.304e6, 2.688e6, 3.072e6, 3.456e6, 3.84e6]
    y_q: Array{Float64}((11,)) [0.0, 384000.0, 768000.0, 1.152e6, 1.536e6, 1.92e6, 2.304e6, 2.688e6, 3.072e6, 3.456e6, 3.84e6]
    halo: Int64 2
    haloη: Int64 1
    halosstx: Int64 1
    halossty: Int64 0
    ep: Int64 0
    x_T_halo: Array{Float64}((12,)) [-192000.0, 192000.0, 576000.0, 960000.0, 1.344e6, 1.728e6, 2.112e6, 2.496e6, 2.88e6, 3.264e6, 3.648e6, 4.032e6]
    y_T_halo: Array{Float64}((12,)) [-192000.0, 192000.0, 576000.0, 960000.0, 1.344e6, 1.728e6, 2.112e6, 2.496e6, 2.88e6, 3.264e6, 3.648e6, 4.032e6]
    x_u_halo: Array{Float64}((13,)) [-384000.0, 0.0, 384000.0, 768000.0, 1.152e6, 1.536e6, 1.92e6, 2.304e6, 2.688e6, 3.072e6, 3.456e6, 3.84e6, 4.224e6]
    y_u_halo: Array{Float64}((14,)) [-576000.0, -192000.0, 192000.0, 576000.0, 960000.0, 1.344e6, 1.728e6, 2.112e6, 2.496e6, 2.88e6, 3.264e6, 3.648e6, 4.032e6, 4.416e6]
    x_v_halo: Array{Float64}((14,)) [-576000.0, -192000.0, 192000.0, 576000.0, 960000.0, 1.344e6, 1.728e6, 2.112e6, 2.496e6, 2.88e6, 3.264e6, 3.648e6, 4.032e6, 4.416e6]
    y_v_halo: Array{Float64}((13,)) [-384000.0, 0.0, 384000.0, 768000.0, 1.152e6, 1.536e6, 1.92e6, 2.304e6, 2.688e6, 3.072e6, 3.456e6, 3.84e6, 4.224e6]
    x_q_halo: Array{Float64}((15,)) [-768000.0, -384000.0, 0.0, 384000.0, 768000.0, 1.152e6, 1.536e6, 1.92e6, 2.304e6, 2.688e6, 3.072e6, 3.456e6, 3.84e6, 4.224e6, 4.608e6]
    y_q_halo: Array{Float64}((15,)) [-768000.0, -384000.0, 0.0, 384000.0, 768000.0, 1.152e6, 1.536e6, 1.92e6, 2.304e6, 2.688e6, 3.072e6, 3.456e6, 3.84e6, 4.224e6, 4.608e6]
    c: Float32 70.035706f0
    dtint: Int64 4934
    nt: Int64 88
    dt: Float32 4934.0f0
    Δt: Float32 0.012848958f0
    Δt_diff: Float32 0.012848958f0
    nadvstep: Int64 389
    nadvstep_half: Int64 194
    dtadvint: Int64 1919326
    dtadvu: Float32 0.078097574f0
    dtadvv: Float32 0.078097574f0
    half_dtadvu: Float32 0.039048787f0
    half_dtadvv: Float32 0.039048787f0
    nout: Int64 17
    nout_total: Int64 6
    t_vec: Array{Int64}((6,)) [0, 4934, 9868, 14802, 19736, 24670]
    f₀: Float64 8.342331420863047e-5
    β: Float64 1.8700492543377578e-11
    f_q: Array{Float32}((11, 11)) Float32[1167.8114 1344.2914 … 2756.1313 2932.6113; 1167.8114 1344.2914 … 2756.1313 2932.6113; … ; 1167.8114 1344.2914 … 2756.1313 2932.6113; 1167.8114 1344.2914 … 2756.1313 2932.6113]
    f_u: Array{Float32}((9, 10)) Float32[19.625803 22.383303 … 41.685802 44.443302; 19.625803 22.383303 … 41.685802 44.443302; … ; 19.625803 22.383303 … 41.685802 44.443302; 19.625803 22.383303 … 41.685802 44.443302]
    f_v: Array{Float32}((10, 9)) Float32[21.004553 23.762053 … 40.307053 43.064552; 21.004553 23.762053 … 40.307053 43.064552; … ; 21.004553 23.762053 … 40.307053 43.064552; 21.004553 23.762053 … 40.307053 43.064552]
  , Main.ShallowWaters.Constants{Float32, Float32}(Float32[0.0021414931, 0.0042829863, 0.0042829863, 0.0021414931], Float32[0.006424479, 0.006424479, 0.012848958], 0.006424479f0, 0.012848958f0, 0.006424479f0, Main.ShallowWaters.SSPRK3coeff{Float32}(5, 25, 16, 7, 0.0006424479f0, 0.44444445f0, 0.5555556f0, 0.0002855324f0), -1.0f0, 9.81f0, -0.06f0, -0.014814815f0, 0.044444446f0, -0.00234375f0, -0.016666668f0, 0.22214422f0, 0.0f0, -1.9910212776572317e-7, 3.9820425553144635e-7, 3.9820425553144635e-7, 64.0f0, 0.015625f0, 32768.0f0), Main.ShallowWaters.Forcing{Float32}(Float32[-17.260805 -13.977639 … 7.0438423 6.041686; -17.260805 -13.977639 … 7.0438423 6.041686; … ; -17.260805 -13.977639 … 7.0438423 6.041686; -17.260805 -13.977639 … 7.0438423 6.041686], Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], Float32[500.0 500.0 … 500.0 500.0; 500.0 500.0 … 500.0 500.0; … ; 500.0 500.0 … 500.0 500.0; 500.0 500.0 … 500.0 500.0], Float32[2.5 2.5 … -2.5 -2.5; 2.5 2.5 … -2.5 -2.5; … ; 2.5 2.5 … -2.5 -2.5; 2.5 2.5 … -2.5 -2.5], Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]), Main.ShallowWaters.PrognosticVars{Float32}(Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0; 0.0 -2.0505126 … -0.42527112 0.0; … ; 0.0 -4.1625385 … 0.6629393 0.0; 0.0 0.0 … 0.0 0.0], Float32[19260.547 24499.795 … -24499.797 -19260.543; 19260.547 24499.795 … -24499.797 -19260.543; … ; -19260.543 -24499.79 … 24499.79 19260.54; -19260.543 -24499.79 … 24499.79 19260.54]), Main.ShallowWaters.DiagnosticVars{Float32, Float32}(Main.ShallowWaters.RungeKuttaVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    u0: Array{Float32}((13, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    u1: Array{Float32}((13, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    v0: Array{Float32}((14, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    v1: Array{Float32}((14, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    η0: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 -2.0505126 … -0.42527112 0.0; … ; 0.0 -4.1625385 … 0.6629393 0.0; 0.0 0.0 … 0.0 0.0]
    η1: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 -0.41382453 … -2.6962225 0.0; … ; 0.0 -1.7205557 … 1.3772216 0.0; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.TendencyVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    du: Array{Float32}((13, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dv: Array{Float32}((14, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dη: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 -122.89504 … 145.37709 0.0; … ; 0.0 -159.46654 … -83.90543 0.0; 0.0 0.0 … 0.0 0.0]
    du_sum: Array{Float32}((13, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dv_sum: Array{Float32}((14, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dη_sum: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    du_comp: Array{Float32}((13, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dv_comp: Array{Float32}((14, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dη_comp: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.VolumeFluxVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    h: Array{Float32}((12, 12)) Float32[500.0 500.0 … 500.0 500.0; 500.0 499.39514 … 498.44037 500.0; … ; 500.0 498.25778 … 500.88565 500.0; 500.0 500.0 … 500.0 500.0]
    h_u: Array{Float32}((11, 12)) Float32[500.0 499.69757 … 499.22018 500.0; 500.0 499.7975 … 499.46753 500.0; … ; 500.0 498.8415 … 499.86316 500.0; 500.0 499.1289 … 500.4428 500.0]
    U: Array{Float32}((11, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 -6.7694473 … -109.07899 0.0; … ; 0.0 -67.811165 … -83.65224 0.0; 0.0 0.0 … 0.0 0.0]
    h_v: Array{Float32}((12, 11)) Float32[500.0 500.0 … 500.0 500.0; 499.69757 499.80658 … 499.1488 499.22018; … ; 499.1289 499.57635 … 500.1455 500.4428; 500.0 500.0 … 500.0 500.0]
    V: Array{Float32}((12, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 129.66449 … 36.298107 0.0; … ; 0.0 91.65537 … -0.25319597 0.0; 0.0 0.0 … 0.0 0.0]
    dUdx: Array{Float32}((10, 12)) Float32[0.0 -6.7694473 … -109.07899 0.0; 0.0 -42.289276 … 71.54745 0.0; … ; 0.0 -15.386147 … -75.07612 0.0; 0.0 67.811165 … 83.65224 0.0]
    dVdy: Array{Float32}((12, 10)) Float32[0.0 0.0 … 0.0 0.0; 129.66449 51.243332 … 9.925919 -36.298107; … ; 91.65537 -88.1035 … 79.17032 0.25319597; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.VorticityVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    h_q: Array{Float32}((11, 11)) Float32[499.8488 499.9033 … 499.5744 499.6101; 499.89874 499.80792 … 500.0908 499.73376; … ; 499.42075 499.9203 … 499.35526 499.93158; 499.56445 499.78818 … 500.07275 500.2214]
    q: Array{Float32}((11, 11)) Float32[2.3363295 2.7223163 … 5.5262747 5.8697996; 2.3378298 2.6671698 … 5.5117674 5.8403783; … ; 2.3557518 2.6877165 … 5.515303 5.8446016; 2.3376591 2.6662288 … 5.5115256 5.8626266]
    q_v: Array{Float32}((10, 11)) Float32[2.3370795 2.6947432 … 5.519021 5.855089; 2.3415656 2.6730022 … 5.5133295 5.8504868; … ; 2.3531442 2.687055 … 5.5154476 5.856292; 2.3467054 2.6769726 … 5.5134144 5.853614]
    U_v: Array{Float32}((10, 11)) Float32[-1.6923618 -15.207882 … -29.945377 -27.269747; -13.957043 -39.318542 … -36.9714 -36.652634; … ; -30.059046 -10.125849 … -48.437702 -23.057089; -16.952791 -7.685052 … -42.262074 -20.91306]
    q_u: Array{Float32}((11, 10)) Float32[2.5293229 2.9032006 … 5.3451467 5.698037; 2.5024998 2.835534 … 5.3337903 5.676073; … ; 2.5217342 2.8641734 … 5.3397207 5.6799526; 2.501944 2.85141 … 5.3473816 5.687076]
    V_u: Array{Float32}((11, 10)) Float32[32.416122 77.64308 … 15.667574 9.074527; 31.106281 75.23631 … -5.160351 -6.000413; … ; 20.868893 -0.070301056 … -23.52471 3.3597667; 22.913843 23.801811 … -19.919178 -0.06329899]
    qhu: Array{Float32}((10, 9)) Float32[-40.981335 -120.32533 … 79.9251 -165.26917; -105.09855 -168.78078 … 168.42824 -203.83551; … ; -27.208714 206.39995 … -166.95721 -267.1556; -20.572674 103.55759 … -183.0807 -233.00833]
    qhv: Array{Float32}((9, 10)) Float32[77.84346 213.33513 … -27.52423 -34.05878; -24.547094 -67.835144 … -267.53098 -231.37555; … ; -61.95896 -227.97551 … 63.041054 69.61436; 52.6258 -0.20135441 … -125.61538 19.083315]
    u_v: Array{Float32}((12, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    v_u: Array{Float32}((13, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dudx: Array{Float32}((12, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dudy: Array{Float32}((13, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dvdx: Array{Float32}((13, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dvdy: Array{Float32}((14, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.BernoulliVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    u²: Array{Float32}((13, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    v²: Array{Float32}((14, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    KEu: Array{Float32}((12, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    KEv: Array{Float32}((14, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    p: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 -378.67166 … -978.3535 0.0; … ; 0.0 -1092.9916 … 556.4993 0.0; 0.0 0.0 … 0.0 0.0]
    dpdx: Array{Float32}((11, 12)) Float32[0.0 -378.67166 … -978.3535 0.0; 0.0 504.29623 … 1290.0183 0.0; … ; 0.0 -732.61304 … 1283.8964 0.0; 0.0 1092.9916 … -556.4993 0.0]
    dpdy: Array{Float32}((12, 11)) Float32[0.0 0.0 … 0.0 0.0; -378.67166 518.90894 … -888.8618 978.3535; … ; -1092.9916 1655.4658 … 928.953 -556.4993; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.BottomdragVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    sqrtKE: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    sqrtKE_u: Array{Float32}((11, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    sqrtKE_v: Array{Float32}((12, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    Bu: Array{Float32}((11, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    Bv: Array{Float32}((12, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.ArakawaHsuVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    qα: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    qβ: Array{Float32}((11, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    qγ: Array{Float32}((11, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    qδ: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.LaplaceVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    Lu: Array{Float32}((11, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    Lv: Array{Float32}((12, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dLudx: Array{Float32}((10, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dLudy: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dLvdx: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dLvdy: Array{Float32}((12, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.SmagorinskyVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    DT: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    DS: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    νSmag: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    DS_q: Array{Float32}((13, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    νSmag_q: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    S12: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    S21: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    S11: Array{Float32}((10, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    S22: Array{Float32}((12, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    LLu1: Array{Float32}((9, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    LLu2: Array{Float32}((11, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    LLv1: Array{Float32}((10, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    LLv2: Array{Float32}((12, 9)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.SemiLagrangeVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    halosstx: Int64 1
    halossty: Int64 0
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    ep: Int64 0
    xd: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    yd: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    um: Array{Float32}((13, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    vm: Array{Float32}((14, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    u_T: Array{Float32}((12, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    um_T: Array{Float32}((12, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    v_T: Array{Float32}((14, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    vm_T: Array{Float32}((14, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    uinterp: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    vinterp: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ssti: Array{Float32}((12, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    sst_ref: Array{Float32}((12, 10)) Float32[19260.547 24499.795 … -24499.797 -19260.543; 19260.547 24499.795 … -24499.797 -19260.543; … ; -19260.543 -24499.79 … 24499.79 19260.54; -19260.543 -24499.79 … 24499.79 19260.54]
    dsst_comp: Array{Float32}((12, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  , Main.ShallowWaters.PrognosticVars{Float32}(Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]), Main.ShallowWaters.ZBVars{Float32}
    nx: Int64 10
    ny: Int64 10
    bc: String "nonperiodic"
    halo: Int64 2
    haloη: Int64 1
    halosstx: Int64 1
    halossty: Int64 0
    nux: Int64 9
    nuy: Int64 10
    nvx: Int64 10
    nvy: Int64 9
    nqx: Int64 11
    nqy: Int64 11
    dudx: Array{Float32}((12, 14)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dudy: Array{Float32}((13, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dvdx: Array{Float32}((13, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dvdy: Array{Float32}((14, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    γ: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    γ_u: Array{Float32}((9, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    γ_v: Array{Float32}((10, 9)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    G: Array{Float32}((3, 3)) Float32[1.0 2.0 1.0; 2.0 4.0 2.0; 1.0 2.0 1.0]
    ζ: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ζsq: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    D: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    Dsq: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    D_n: Array{Float32}((13, 13)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    D_nT: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    D_q: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    Dhat: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    Dhatsq: Array{Float32}((12, 12)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    Dhatq: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ζpDT: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ζsqT: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ζD: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ζDT: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ζDhat: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    trace: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ζD_filtered: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    ζDhat_filtered: Array{Float32}((11, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    trace_filtered: Array{Float32}((10, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dζDdx: Array{Float32}((9, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dζDhatdy: Array{Float32}((11, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dtracedx: Array{Float32}((9, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    S_u: Array{Float32}((9, 10)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dζDhatdx: Array{Float32}((10, 11)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dζDdy: Array{Float32}((10, 9)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    dtracedy: Array{Float32}((10, 9)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    S_v: Array{Float32}((10, 9)) Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  ), 0)

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