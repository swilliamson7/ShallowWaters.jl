using NetCDF, Parameters, Printf, Dates, Interpolations

using Enzyme#main

@with_kw struct Parameter

    T=Float32                 # number format

    Tprog=T                   # number format for prognostic variables
    Tcomm=Tprog               # number format for ghost-point copies
    Tini=Tprog                # number format to reduce precision for initial conditions

    # DOMAIN RESOLUTION AND RATIO
    nx::Int=128                         # number of grid cells in x-direction
    Lx::Real=3840e3                     # length of the domain in x-direction [m]
    L_ratio::Real=1                     # Domain aspect ratio of Lx/Ly

    # PHYSICAL CONSTANTS
    g::Real=9.81                        # gravitational acceleration [m/s] ##Changed
    H::Real=500.                        # layer thickness at rest [m]
    ρ::Real=1e3                         # water density [kg/m^3]
    ϕ::Real=45.                         # central latitude of the domain (for coriolis) [°]
    ω::Real=2π/(24*3600)                # Earth's angular frequency [s^-1]
    R::Real=6.371e6                     # Earth's radius [m]

    # SCALE
    scale::Real=2^6                     # multiplicative scale for the momentum equations u,v
    scale_sst::Real=2^15                # multiplicative scale for sst

    # WIND FORCING OPTIONS
    wind_forcing_x::String="double_gyre"      # "channel", "double_gyre", "shear","constant" or "none"
    wind_forcing_y::String="constant"   # "channel", "double_gyre", "shear","constant" or "none"
    Fx0::Real=0.12                      # wind stress strength [Pa] in x-direction
    Fy0::Real=0.0                       # wind stress strength [Pa] in y-direction
    seasonal_wind_x::Bool=true          # Change the wind stress with a sine of frequency ωFx,ωFy
    seasonal_wind_y::Bool=false         # same for y-component
    ωFx::Real=2                         # frequency [1/year] for x component
    ωFy::Real=2                         # frequency [1/year] for y component

    # BOTTOM TOPOGRAPHY OPTIONS
    topography::String="flat"         # "ridge", "seamount", "flat", "ridges", "bathtub"
    topo_ridges_positions::Vector = [0.05,0.25,0.45,0.9]
    topo_height::Real=100.               # height of seamount [m]
    topo_width::Real=300e3              # horizontal scale [m] of the seamount

    # SURFACE RELAXATION
    surface_relax::Bool=false           # yes?
    t_relax::Real=100.                  # time scale of the relaxation [days]
    η_refh::Real=5.                     # height difference [m] of the interface relaxation profile
    η_refw::Real=50e3                   # width [m] of the tangent used for the interface relaxation

    # SURFACE FORCING
    surface_forcing::Bool=false         # yes?
    ωFη::Real=1.0                       # frequency [1/year] for surfance forcing
    A::Real=3e-5                        # Amplitude [m/s]
    ϕk::Real=ϕ                          # Central latitude of Kelvin wave pumping
    wk::Real=10e3                       # width [m] in y of Gaussian used for surface forcing

    # TIME STEPPING OPTIONS
    time_scheme::String="RK"            # Runge-Kutta ("RK") or strong-stability preserving RK
                                        # "SSPRK2","SSPRK3","4SSPRK3"
    RKo::Int=4                          # Order of the RK time stepping scheme (2, 3 or 4)
    RKs::Int=3                          # Number of stages for SSPRK2
    RKn::Int=5                          # n^2 = s = Number of stages  for SSPRK3
    cfl::Real=0.9                       # CFL number (1.0 recommended for RK4, 0.6 for RK3)
    Ndays::Real=5                       # number of days to integrate for
    nstep_diff::Int=1                   # diffusive part every nstep_diff time steps.
    nstep_advcor::Int=0                 # advection and coriolis update every nstep_advcor time steps.
                                        # 0 means it is included in every RK4 substep
    compensated::Bool=false             # Compensated summation in the time integration?

    # BOUNDARY CONDITION OPTIONS
    bc::String="periodic"               # "periodic" or anything else for nonperiodic
    α::Real=2.                          # lateral boundary condition parameter
                                        # 0 free-slip, 0<α<2 partial-slip, 2 no-slip

    # MOMENTUM ADVECTION OPTIONS
    adv_scheme::String="Sadourny"       # "Sadourny" or "ArakawaHsu"
    dynamics::String="nonlinear"        # "linear" or "nonlinear"

    # BOTTOM FRICTION OPTIONS
    bottom_drag::String="quadratic"     # "linear", "quadratic" or "none"
    cD::Real=1e-5                       # bottom drag coefficient [dimensionless] for quadratic
    τD::Real=300.                       # bottom drag coefficient [days] for linear

    # DIFFUSION OPTIONS
    diffusion::String="constant"        # "Smagorinsky" or "constant", biharmonic in both cases
    νB::Real=500.0                      # [m^2/s] scaling constant for constant biharmonic diffusion
    cSmag::Real=0.15                    # Smagorinsky coefficient [dimensionless]

    # TRACER ADVECTION
    tracer_advection::Bool=true         # yes?
    tracer_relaxation::Bool=true        # yes?
    tracer_consumption::Bool=false      # yes?
    sst_initial::String="waves"         # "west", "south", "linear", "waves","rect", "flat" or "restart"
    sst_rect_coords::Array{Float64,1}=[0.,0.15,0.,1.0]
                                        # (x0,x1,y0,y1) are the size of the rectangle in [0,1]
    Uadv::Real=0.2                      # Velocity scale [m/s] for tracer advection
    SSTmax::Real=1.                     # tracer (sea surface temperature) max for initial conditions
    SSTmin::Real=-1.                    # tracer (sea surface temperature) min for initial conditions
    τSST::Real=100                      # tracer restoring time scale [days]
    jSST::Real=365                      # tracer consumption [days]
    SSTw::Real=5e5                      # width [m] of the tangent used for the IC and interface relaxation
    SSTϕ::Real=0.5                      # latitude/longitude fraction ∈ [0,1] of sst edge
    SSTwaves_ny::Real=4                 # wave crests/troughs in y
    SSTwaves_nx::Real=SSTwaves_ny*L_ratio  # wave crests/troughs in x
    SSTwaves_p::Real=1/2                # power for rectangles (p<1)/smootheness(p>=1) of waves

    # OUTPUT OPTIONS
    output::Bool=false                  # netcdf output?
    output_vars::Array{String,1}=["u","v","η","sst"]  # which variables to output? "du","dv","dη" also allowed
    output_dt::Real=24                  # output time step [hours]
    outpath::String=pwd()               # path to output folder
    compression_level::Int=3            # compression level
    return_time::Bool=false             # return time of simulation of progn vars?

    # INITIAL CONDITIONS
    initial_cond::String="rest"         # "rest" or "ncfile" for restart from file
    initpath::String=outpath            # folder where to pick the restart files from
    init_run_id::Int=0                  # run id for restart from run number
    init_starti::Int=-1                 # timestep to start from (-1 meaning last)
    get_id_mode::String="continue"      # How to determine the run id: "continue" or "fill"
    run_id::Int=-1                      # Output with a specific run id
    init_interpolation::Bool=true       # Interpolate the initial conditions in case grids don't match?

end

@with_kw struct Grid{T<:AbstractFloat,Tprog<:AbstractFloat}

    # Parameters taken from Parameter struct
    nx::Int                             # number of grid cells in x-direction
    Lx::Real                            # length of the domain in x-direction [m]
    L_ratio::Real                       # Domain aspect ratio of Lx/Ly
    bc::String                          # boundary condition, "periodic" or "nonperiodic"
    g::Real                             # gravitational acceleration [m/s]
    H::Real                             # layer thickness at rest [m]
    cfl::Real                           # CFL number (1.0 recommended for RK4, 0.6 for RK3)
    Ndays::Real                         # number of days to integrate for
    nstep_diff::Int                     # diffusive terms every nstep_diff time steps.
    nstep_advcor::Int                   # nonlinear terms every nstep_advcor time steps.
    Uadv::Real                          # Velocity scale [m/s] for tracer advection
    output_dt::Real                     # output time step [hours]
    ω::Real                             # Earth's angular frequency [s^-1]
    ϕ::Real                             # central latitue of the domain (for coriolis) [°]
    R::Real                             # Earth's radius [m]
    scale::Real                         # multiplicative scale for momentum equations [1]

    # DOMAIN SIZES
    Δ::Real=Lx / nx                         # grid spacing
    ny::Int=Int(round(Lx / L_ratio / Δ))    # number of grid cells in y-direction
    Ly::Real=ny * Δ                         # length of domain in y-direction

    # NUMBER OF GRID POINTS
    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # TOTAL NUMBER OF GRID POINTS
    nT::Int = nx*ny                     # T-grid
    nu::Int = nux*nuy                   # u-grid
    nv::Int = nvx*nvy                   # v-grid
    nq::Int = nqx*nqy                   # q-grid

    # GRID VECTORS
    x_T::AbstractVector = Δ*Array(1:nx) .- Δ/2
    y_T::AbstractVector = Δ*Array(1:ny) .- Δ/2
    x_u::AbstractVector = if (bc == "periodic") Δ*Array(0:nx-1) else Δ*Array(1:nx-1) end
    y_u::AbstractVector = y_T
    x_v::AbstractVector = x_T
    y_v::AbstractVector = Δ*Array(1:ny-1)
    x_q::AbstractVector = if bc == "periodic" x_u else Δ*Array(1:nx+1) .- Δ end
    y_q::AbstractVector = Δ*Array(1:ny+1) .- Δ

    # HALO SIZES
    halo::Int=2                         # halo size for u,v (Biharmonic stencil requires 2)
    haloη::Int=1                        # halo size for η
    halosstx::Int=1                     # halo size for tracer sst in x
    halossty::Int=0                     # halo size for tracer sst in y

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end  # is there a u-point on the left edge?

    # GRID VECTORS WITH HALO
    x_T_halo::AbstractVector = Δ*Array(0:nx+1) .- Δ/2
    y_T_halo::AbstractVector = Δ*Array(0:ny+1) .- Δ/2
    x_u_halo::AbstractVector = if (bc == "periodic") Δ*Array(-2:nx+1) else Δ*Array(-1:nx+1) end
    y_u_halo::AbstractVector = Δ*Array(-1:ny+2) .- Δ/2
    x_v_halo::AbstractVector = Δ*Array(-1:nx+2) .- Δ/2
    y_v_halo::AbstractVector = Δ*Array(-1:ny+1)
    x_q_halo::AbstractVector = if bc == "periodic" x_u_halo else Δ*Array(-1:nx+3) .- Δ end
    y_q_halo::AbstractVector = Δ*Array(-1:ny+3) .- Δ

    # TIME STEPS
    c::Real = √(g*H)                            # shallow water gravity wave speed
    dtint::Int = Int(floor(cfl*Δ/c))            # dt converted to Int
    nt::Int = Int(ceil(Ndays*3600*24/dtint))    # number of time steps to integrate
    dt::T = T(dtint)                            # time step [s]
    Δt::T = T(dtint/Δ)                          # time step divided by grid spacing [s/m]
    Δt_diff::Tprog = Tprog(nstep_diff*dtint/Δ)  # time step for diffusive terms

    # TIME STEPS FOR ADVECTION
    nadvstep::Int = max(1,Int(floor(Δ/Uadv/dtint)))         # advection each n time steps
    nadvstep_half::Int = nadvstep ÷ 2                       # nadvstep ÷ 2
    dtadvint::Int = nadvstep*dtint                          # advection time step [s]
    # divide by scale here to undo the scaling in u,v for tracer advection
    dtadvu::T = T(dtadvint*nx/Lx/scale)                     # Rescaled advection time step for u [s/m]
    dtadvv::T = T(dtadvint*ny/Ly/scale)                     # Rescaled advection time step for v [s/m]
    half_dtadvu::T = T(dtadvint*nx/Lx/2/scale)              # dtadvu/2
    half_dtadvv::T = T(dtadvint*ny/Ly/2/scale)              # dtadvv/2

    # N TIME STEPS FOR OUTPUT
    nout::Int = max(1,Int(floor(output_dt*3600/dtint)))     # output every n time steps
    nout_total::Int = (nt ÷ nout)+1                         # total number of output time steps
    t_vec::AbstractVector = Array(0:nout_total-1)*dtint     # time vector

    # CORIOLIS
    f₀::Float64 = coriolis_at_lat(ω,ϕ)                      # Coriolis parameter
    β::Float64 = β_at_lat(ω,R,ϕ)                            # Derivate of Coriolis parameter wrt latitude
    # scale only f_q as it's used for non-linear advection
    f_q::Array{T,2} = T.(scale*Δ*(f₀ .+ β*(yy_q(bc,x_q_halo,y_q_halo) .- Ly/2)))  # same on the q-grid
    # f_u, f_v are only used for linear dynamics (scaling implicit)
    f_u::Array{T,2} = T.(Δ*(f₀ .+ β*(meshgrid(x_u,y_u)[2] .- Ly/2)))        # f = f₀ + βy on the u-grid
    f_v::Array{T,2} = T.(Δ*(f₀ .+ β*(meshgrid(x_v,y_v)[2] .- Ly/2)))        # same on the v-grid
end

"""Helper function to create yy_q based on the boundary condition bc."""
function yy_q(bc::String,x_q_halo::AbstractVector,y_q_halo::AbstractVector)
    if bc == "periodic"
        # points on the right edge needed too
        _,yy_q = meshgrid(x_q_halo[3:end-1],y_q_halo[3:end-2])
    else
        _,yy_q = meshgrid(x_q_halo[3:end-2],y_q_halo[3:end-2])
    end

    return yy_q
end


"""Generator function for the Grid struct."""
function Grid{T,Tprog}(P::Parameter) where {T<:AbstractFloat,Tprog<:AbstractFloat}
    @unpack nx,Lx,L_ratio = P
    @unpack bc,g,H,cfl = P
    @unpack Ndays,nstep_diff,nstep_advcor = P
    @unpack Uadv,output_dt = P
    @unpack ϕ,ω,R,scale = P

    return Grid{T,Tprog}(nx=nx,Lx=Lx,L_ratio=L_ratio,bc=bc,g=g,H=H,cfl=cfl,Ndays=Ndays,
                nstep_diff=nstep_diff,nstep_advcor=nstep_advcor,Uadv=Uadv,output_dt=output_dt,
                ϕ=ϕ,ω=ω,R=R,scale=scale)
end

"""Meter per 1 degree of latitude (or longitude at the equator)."""
function m_per_lat(R::Real)
    return 2π*R/360.
end

"""Coriolis parameter f [1/s] at latitude ϕ [°] given Earth's rotation ω [1/s]."""
function coriolis_at_lat(ω::Real,ϕ::Real)
    return 2*ω*sind(ϕ)
end

"""Coriolis parameter's derivative β wrt latitude [(ms)^-1] at latitude ϕ, given
Earth's rotation ω [1/s] and radius R [m]."""
function β_at_lat(ω::Real,R::Real,ϕ::Real)
    return 2*ω/R*cosd(ϕ)
end

"""Similar to the numpy meshgrid function:
repeats x length(y)-times and vice versa. Returns two matrices xx,yy of same shape so that
each row of xx is x and each column of yy is y."""
function meshgrid(x::AbstractVector,y::AbstractVector)
    m,n = length(x),length(y)

    # preallocate preserving the data type of x,y
    xx = zeros(eltype(x),m,n)
    yy = zeros(eltype(y),m,n)

    for i in 1:m
        xx[i,:] .= x[i]
    end

    for i in 1:n
        yy[:,i] .= y[i]
    end

    return xx,yy
end

struct Constants{T<:AbstractFloat,Tprog<:AbstractFloat}

    # RUNGE-KUTTA COEFFICIENTS 2nd/3rd/4th order including timestep Δt
    RKaΔt::Array{Tprog,1}
    RKbΔt::Array{Tprog,1}
    Δt_Δs::Tprog            # Δt/(s-1) wher s the number of stages
    Δt_Δ::Tprog             # Δt/Δ - timestep divided by grid spacing
    Δt_Δ_half::Tprog        # 1/2 * Δt/Δ

    # BOUNDARY CONDITIONS
    one_minus_α::Tprog      # tangential boundary condition for the ghost-point copy

    # PHYSICAL CONSTANTS
    g::T                    # gravity
    cD::T                   # quadratic bottom friction - incl grid spacing
    rD::T                   # linear bottom friction - incl grid spacing
    γ::T                    # frequency of interface relaxation
    cSmag::T                # Smagorinsky constant
    νB::T                   # biharmonic diffusion coefficient
    τSST::T                 # tracer restoring timescale
    jSST::T                 # tracer consumption timescale
    ωFη::Float64            # frequency [1/s] of seasonal surface forcing incl 2π
    ωFx::Float64            # frequency [1/s] of seasonal wind x incl 2π
    ωFy::Float64            # frequency [1/2] of seasonal wind y incl 2π

    # SCALING
    scale::T                # multiplicative constant for low-precision arithmetics
    scale_inv::T            # and its inverse
    scale_sst::T            # scale for sst
end

"""Generator function for the mutable struct Constants."""
function Constants{T,Tprog}(P::Parameter,G::Grid) where {T<:AbstractFloat,Tprog<:AbstractFloat}

    # Runge-Kutta 2nd/3rd/4th order coefficients including time step Δt and grid spacing Δ
    # a are the coefficents to sum the rhs on the fly, such that sum=1
    # b are the coefficents for the update that used for a new evaluation of the RHS
    if P.RKo == 2     # Heun's method
        RKaΔt = Tprog.([1/2,1/2]*G.dtint/G.Δ)
        RKbΔt = Tprog.([1]*G.dtint/G.Δ)
    elseif P.RKo == 3     # version 2 / Heun's 3rd order
        RKaΔt = Tprog.([1/4,0.,3/4]*G.dtint/G.Δ)
        RKbΔt = Tprog.([1/3,2/3]*G.dtint/G.Δ)
    elseif P.RKo == 4
        RKaΔt = Tprog.([1/6,1/3,1/3,1/6]*G.dtint/G.Δ)
        RKbΔt = Tprog.([.5,.5,1.]*G.dtint/G.Δ)
    end

    # Δt/(s-1) for SSPRK2
    Δt_Δs = convert(Tprog,G.dtint/G.Δ/(P.RKs-1))

    # time step and half the time step including the grid spacing as this is not included in the RHS
    Δt_Δ = convert(Tprog,G.dtint/G.Δ)
    Δt_Δ_half = convert(Tprog,G.dtint/G.Δ/2)


    # BOUNDARY CONDITIONS AND PHYSICS
    one_minus_α = convert(Tprog,1-P.α)      # for the ghost point copy/tangential boundary conditions
    g = convert(T,P.g)                      # gravity - for Bernoulli potential

    # BOTTOM FRICTION COEFFICIENTS
    # incl grid spacing Δ for non-dimensional gradients
    # include scale for quadratic cD only to unscale the scale^2 in u^2
    cD = convert(T,-G.Δ*P.cD/P.scale)     # quadratic drag [m]
    rD = convert(T,-G.Δ/(P.τD*24*3600))   # linear drag [m/s]

    # INTERFACE RELAXATION FREQUENCY
    # incl grid spacing Δ for non-dimensional gradients
    γ = convert(T,G.Δ/(P.t_relax*3600*24))    # [m/s]

    # BIHARMONIC DIFFUSION
    # undo scaling here as smagorinksy diffusion contains scale^2 due to ~u^2
    cSmag = convert(T,-P.cSmag/P.scale)   # Smagorinsky coefficient
    νB = convert(T,-P.νB/30000)           # linear scaling based on 540m^s/s at Δ=30km

    # TRACER ADVECTION
    τSST = convert(T,G.dtadvint/(P.τSST*3600*24))   # tracer restoring [1]
    jSST = convert(T,G.dtadvint/(P.jSST*3600*24))   # tracer consumption [1]

    @unpack tracer_relaxation, tracer_consumption = P
    τSST = tracer_relaxation ? τSST : zero(T)       # set zero as τ,j will be added   
    jSST = tracer_consumption ? jSST : zero(T)      # and executed in one loop

    # TIME DEPENDENT FORCING
    ωFη = -2π*P.ωFη/24/365.25/3600
    ωFx = 2π*P.ωFx/24/365.25/3600
    ωFy = 2π*P.ωFy/24/365.25/3600

    # SCALING
    scale = convert(T,P.scale)
    scale_inv = convert(T,1/P.scale)
    scale_sst = convert(T,P.scale_sst)

    return Constants{T,Tprog}(  RKaΔt,RKbΔt,Δt_Δs,Δt_Δ,Δt_Δ_half,
                                one_minus_α,
                                g,cD,rD,γ,cSmag,νB,τSST,jSST,
                                ωFη,ωFx,ωFy,
                                scale,scale_inv,scale_sst)
end

struct Forcing{T<:AbstractFloat}
    Fx::Array{T,2}
    Fy::Array{T,2}
    H::Array{T,2}
    η_ref::Array{T,2}
    Fη::Array{T,2}
    #sst_ref::Array{T,2}
    #sst_γ::Array{T,2}
end

function Forcing{T}(P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack wind_forcing_x,wind_forcing_y,topography = P

    if wind_forcing_x == "channel"
        Fx,_ = ChannelWind(T,P,G)
    elseif wind_forcing_x == "shear"
        Fx,_ = ShearWind(T,P,G)
    elseif wind_forcing_x == "double_gyre"
        Fx,_ = DoubleGyreWind(T,P,G)
    elseif wind_forcing_x == "constant"
        Fx,_ = ConstantWind(T,P,G)
    elseif wind_forcing_x == "none"
        Fx,_ = NoWind(T,P,G)
    end

    if wind_forcing_y == "channel"
        _,Fy = ChannelWind(T,P,G)
    elseif wind_forcing_y == "shear"
        _,Fy = ShearWind(T,P,G)
    elseif wind_forcing_y == "double_gyre"
        _,Fy = DoubleGyreWind(T,P,G)
    elseif wind_forcing_y == "constant"
        _,Fy = ConstantWind(T,P,G)
    elseif wind_forcing_x == "none"
        _,Fy = NoWind(T,P,G)
    end

    if topography == "ridge"
        H,_ = Ridge(T,P,G)
    elseif topography == "ridges"
        H = Ridges(T,P,G)
    elseif topography == "seamount"
        H = Seamount(T,P,G)
    elseif topography == "flat"
        H = FlatBottom(T,P,G)
    end

    η_ref = InterfaceRelaxation(T,P,G)
    Fη = KelvinPump(T,P,G)

    return Forcing{T}(Fx,Fy,H,η_ref,Fη)
end

"""Returns the constant forcing matrices Fx,Fy that vary only meriodionally/zonally
as a cosine with strongest forcing in the middle and vanishing forcing at boundaries."""
function ChannelWind(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack Δ,x_u,y_u,Lx,Ly = G
    @unpack Fx0,Fy0,H,ρ,scale = P

    # for non-dimensional gradients the wind forcing needs to contain the grid spacing Δ
    xx_u,yy_u = meshgrid(x_u,y_u)
    Fx = (scale*Δ*Fx0/ρ/H)*cos.(π*(yy_u/Ly .- 1/2)).^2
    Fy = (scale*Δ*Fy0/ρ/H)*cos.(π*(xx_u/Lx .- 1/2)).^2

    return T.(Fx),T.(Fy)
end

"""Returns the constant forcing matrices Fx,Fy that vary only meriodionally/zonally
as a hyperbolic tangent with strongest shear in the middle."""
function ShearWind(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack Δ,x_u,y_u,Lx,Ly = G
    @unpack Fx0,Fy0,H,ρ,scale = P

    # for non-dimensional gradients the wind forcing needs to contain the grid spacing Δ
    xx_u,yy_u = meshgrid(x_u,y_u)
    Fx = (scale*Δ*Fx0/ρ/H)*tanh.(2π*(yy_u/Ly .- 1/2))
    Fy = (scale*Δ*Fy0/ρ/H)*tanh.(2π*(xx_u/Lx .- 1/2))

    return T.(Fx),T.(Fy)
end

"""Returns the constant forcing matrices Fx,Fy that vary only meriodionally/zonally
with a superposition of sin & cos for a double gyre circulation.
See Cooper&Zanna 2015 or Kloewer et al 2018."""
function DoubleGyreWind(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack Δ,x_u,y_u,Lx,Ly = G
    @unpack Fx0,Fy0,H,ρ,scale = P

    # for non-dimensional gradients the wind forcing needs to contain the grid spacing Δ
    xx_u,yy_u = meshgrid(x_u,y_u)
    Fx = (scale*Δ*Fx0/ρ/H)*(cos.(2π*(yy_u/Ly .- 1/2)) + 2*sin.(π*(yy_u/Ly .- 1/2)))
    Fy = (scale*Δ*Fy0/ρ/H)*(cos.(2π*(xx_u/Lx .- 1/2)) + 2*sin.(π*(xx_u/Lx .- 1/2)))
    return T.(Fx),T.(Fy)
end

"""Returns constant in in space forcing matrices Fx,Fy."""
function ConstantWind(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack Δ,nux,nuy,nvx,nvy = G
    @unpack Fx0,Fy0,H,ρ,scale = P

    # for non-dimensional gradients the wind forcing needs to contain the grid spacing Δ
    Fx = T.(scale*Δ*Fx0/ρ/H)*ones(T,nux,nuy)
    Fy = T.(scale*Δ*Fy0/ρ/H)*ones(T,nvx,nvy)

    return Fx,Fy
end

"""Returns constant in in space forcing matrices Fx,Fy."""
function NoWind(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack nux,nuy,nvx,nvy = G

    # for non-dimensional gradients the wind forcing needs to contain the grid spacing Δ
    Fx = zeros(T,nux,nuy)
    Fy = zeros(T,nvx,nvy)

    return Fx,Fy
end

"""Returns a reference state for Newtonian cooling/surface relaxation shaped as a
hyperbolic tangent to force the continuity equation."""
function InterfaceRelaxation(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack x_T,y_T,Ly,Lx = G
    @unpack η_refh,η_refw = P

    # width of a tangent is defined as the distance between 65% of its minimum value and 65% of the max
    xx_T,yy_T = meshgrid(x_T,y_T)
    η_ref = -(η_refh/2)*tanh.(2π*(Ly/(4*η_refw))*(yy_T/Ly .- 1/2))
    return T.(η_ref)
end

"""Returns a matrix of water depth for the whole domain that contains a
Gaussian seamount in the middle. Water depth, heigth and width of the
seamount are adjusted with the constants H, topofeat_height and topofeat_width."""
function Seamount(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack x_T_halo,y_T_halo,Lx,Ly = G
    @unpack topo_width,topo_height,H = P

    xx_T,yy_T = meshgrid(x_T_halo,y_T_halo)
    bumpx = exp.(-((xx_T .- Lx/2).^2)/(2*topo_width^2))
    bumpy = exp.(-((yy_T .- Ly/2).^2)/(2*topo_width^2))

    return T.(H .- topo_height*bumpx.*bumpy)
end

"""Returns a matrix of water depth for the whole domain that contains a
meridional Gaussian ridge in the middle. Water depth, heigth and width of the
ridge are adjusted with the constants water_depth, topofeat_height and topofeat_width."""
function Ridge(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack x_T_halo,y_T_halo,Lx,Ly = G
    @unpack topo_width,topo_height,H = P

    xx_T,yy_T = meshgrid(x_T_halo,y_T_halo)
    bumpx = exp.(-((xx_T .- Lx/2).^2)/(2*topo_width^2))
    bumpy = exp.(-((yy_T .- Ly/2).^2)/(2*topo_width^2))

    Hx = H .- topo_height*bumpx
    Hy = H .- topo_height*bumpy
    return T.(Hx),T.(Hy)
end

"""Same as Ridge() but for n ridges at various x positions."""
function Ridges(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack x_T_halo,y_T_halo,Lx,Ly = G
    @unpack topo_width,topo_height,H = P
    @unpack topo_ridges_positions = P
    n_ridges = length(topo_ridges_positions)

    xx_T,yy_T = meshgrid(x_T_halo,y_T_halo)

    # loop over bumps in x direction
    R = zero(xx_T) .+ H

    for i in 1:n_ridges
        R .-= topo_height*exp.(-((xx_T .- topo_ridges_positions[i]*Lx).^2)/(2*topo_width^2))
    end

    return T.(R)
end

"""Returns a matrix of constant water depth H."""
function FlatBottom(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}
    @unpack nx,ny,haloη = G
    @unpack H = P
    return fill(T(H),(nx+2*haloη,ny+2*haloη))
end

"""Returns Kelvin wave pumping forcing of the continuity equation."""
function KelvinPump(::Type{T},P::Parameter,G::Grid) where {T<:AbstractFloat}

    @unpack x_T,y_T,Lx,Ly = G
    @unpack R,ϕ,Δ = G
    @unpack A,ϕk,wk = P

    xx_T,yy_T = meshgrid(x_T,y_T)

    mϕ = 2π*R/360.          # meters per degree latitude
    y0 = Ly/2 - (ϕ-ϕk)*mϕ   # y[m] for central latitude of pumping

    Fη = A*Δ*exp.(-(yy_T.-y0).^2/(2*wk^2))

    return T.(Fη)
end

"""Time evolution of forcing."""
function Ftime(::Type{T},t::Int,ω::Real) where {T<:AbstractFloat}
    return convert(T,sin(ω*t))
end

"""Runge Kutta time stepping scheme diagnostic cariables collected in a struct."""
@with_kw struct RungeKuttaVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end  # is there a u-point on the left edge?

    u0::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)     # u-velocities for RK updates
    u1::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)
    v0::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)     # v-velocities for RK updates
    v1::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)
    η0::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)     # sea surface height for RK updates
    η1::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)
end

"""Generator function for RungeKutta VarCollection."""
function RungeKuttaVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return RungeKuttaVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###################################################

"""Tendencies collected in a struct."""
@with_kw struct TendencyVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end  # is there a u-point on the left edge?

    du::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)     # tendency of u without time step
    dv::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)     # tendency of v without time step
    dη::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)     # tendency of η without time step

    # sum of tendencies (incl time step) over all sub-steps
    du_sum::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo) 
    dv_sum::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)
    dη_sum::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)

    # compensation for tendencies (variant of Kahan summation)
    du_comp::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo) 
    dv_comp::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)
    dη_comp::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)
end

"""Generator function for Tendencies VarCollection."""
function TendencyVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return TendencyVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###########################################################

"""VolumeFluxes collected in a struct."""
@with_kw struct VolumeFluxVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    h::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)         # layer thickness
    h_u::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)     # layer thickness on u-grid
    U::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)       # U=uh volume flux

    h_v::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)     # layer thickness on v-grid
    V::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)       # V=vh volume flux

    dUdx::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη)    # gradients thereof
    dVdy::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-2)
end

"""Generator function for VolumeFluxes VarCollection."""
function VolumeFluxVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return VolumeFluxVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###############################################################

"""Vorticity variables collected in a struct."""
@with_kw struct VorticityVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    h_q::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)  # layer thickness h interpolated on q-grid
    q::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)    # potential vorticity

    q_v::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη-1)  # q interpolated on v-grid
    U_v::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη-1)  # mass flux U=uh on v-grid

    q_u::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-2)  # q interpolated on u-grid
    V_u::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-2)  # mass flux V=vh on v-grid

    qhu::Array{T,2} = zeros(T,nvx,nvy)            # potential vorticity advection term u-component
    qhv::Array{T,2} = zeros(T,nux,nuy)            # potential vorticity advection term v-component

    u_v::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo-1)  # u-velocity on v-grid
    v_u::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo-1)  # v-velocity on u-grid

    dudx::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)   # ∂u/∂x
    dudy::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo-1)   # ∂u/∂y

    dvdx::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo)   # ∂v/∂x
    dvdy::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)   # ∂v/∂y
end

"""Generator function for Vorticity VarCollection."""
function VorticityVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return VorticityVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Bernoulli variables collected in a struct."""
@with_kw struct BernoulliVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    u²::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)         # u-velocity squared
    v²::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)         # v-velocity squared

    KEu::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)      # u-velocity squared on T-grid
    KEv::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)      # v-velocity squared on T-grid

    p::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)          # Bernoulli potential
    dpdx::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)     # ∂p/∂x
    dpdy::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)     # ∂p/∂y
end

"""Generator function for Bernoulli VarCollection."""
function BernoulliVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return BernoulliVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Bottomdrag variables collected in a struct."""
@with_kw struct BottomdragVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    sqrtKE::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)       # sqrt of kinetic energy
    sqrtKE_u::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)   # interpolated on u-grid
    sqrtKE_v::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)   # interpolated on v-grid

    Bu::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)         # bottom friction term u-component
    Bv::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)         # bottom friction term v-component
end

"""Generator function for Bottomdrag VarCollection."""
function BottomdragVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return BottomdragVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""ArakawaHsu variables collected in a struct."""
@with_kw struct ArakawaHsuVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    # Linear combination of potential vorticity
    qα::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη-2)
    qβ::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-2)
    qγ::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-2)
    qδ::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη-2)
end

"""Generator function for ArakawaHsu VarCollection."""
function ArakawaHsuVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return ArakawaHsuVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Laplace variables collected in a struct."""
@with_kw struct LaplaceVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    Lu::Array{T,2} = zeros(T,nux+2*halo-2,nuy+2*halo-2)         # ∇²u
    Lv::Array{T,2} = zeros(T,nvx+2*halo-2,nvy+2*halo-2)         # ∇²v

    # Derivatives of Lu,Lv
    dLudx::Array{T,2} = zeros(T,nux+2*halo-3,nuy+2*halo-2)
    dLudy::Array{T,2} = zeros(T,nux+2*halo-2,nuy+2*halo-3)
    dLvdx::Array{T,2} = zeros(T,nvx+2*halo-3,nvy+2*halo-2)
    dLvdy::Array{T,2} = zeros(T,nvx+2*halo-2,nvy+2*halo-3)
end

"""Generator function for Laplace VarCollection."""
function LaplaceVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return LaplaceVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Smagorinsky variables collected in a struct."""
@with_kw struct SmagorinskyVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    DT::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)       # Tension squared (on the T-grid)
    DS::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)       # Shearing strain squared (on the T-grid)
    νSmag::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)    # Viscosity coefficient

    # Tension squared on the q-grid
    DS_q::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo)

    # Smagorinsky viscosity coefficient on the q-grid
    νSmag_q::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)

    # Entries of the Smagorinsky viscous tensor
    S12::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)
    S21::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)

    S11::Array{T,2} = zeros(T,nux+2*halo-3,nuy+2*halo-2)
    S22::Array{T,2} = zeros(T,nvx+2*halo-2,nvy+2*halo-3)

    # u- and v-components 1 and 2 of the biharmonic diffusion tendencies
    LLu1::Array{T,2} = zeros(T,nux+2*halo-4,nuy+2*halo-2)
    LLu2::Array{T,2} = zeros(T,nx+1,ny)

    LLv1::Array{T,2} = zeros(T,nx,ny+1)
    LLv2::Array{T,2} = zeros(T,nvx+2*halo-2,nvy+2*halo-4)
end

"""Generator function for Smagorinsky VarCollection."""
function SmagorinskyVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return SmagorinskyVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""SemiLagrange variables collected in a struct."""
@with_kw struct SemiLagrangeVars{T<:AbstractFloat}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int
    halosstx::Int
    halossty::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    xd::Array{T,2} = zeros(T,nx,ny)                         # departure points x-coord
    yd::Array{T,2} = zeros(T,nx,ny)                         # departure points y-coord

    um::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)         # u-velocity temporal mid-point
    vm::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)         # v-velocity temporal mid-point

    u_T::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)      # u-velocity interpolated on T-grid
    um_T::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)     # um interpolated on T-grid
    v_T::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)      # v-velocity interpolated on T-grid
    vm_T::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)     # vm interpolated on T-grid

    uinterp::Array{T,2} = zeros(T,nx,ny)                    # u interpolated on mid-point xd,yd
    vinterp::Array{T,2} = zeros(T,nx,ny)                    # v interpolated on mid-point xd,yd

    ssti::Array{T,2} = zeros(T,nx+2*halosstx,ny+2*halossty) # sst interpolated on departure points
    sst_ref::Array{T,2} = zeros(T,nx+2*halosstx,ny+2*halossty) # sst initial conditions for relaxation

    # compensated summation
    dsst_comp::Array{T,2} = zeros(T,nx+2*halosstx,ny+2*halossty)
end

"""Generator function for SemiLagrange VarCollection."""
function SemiLagrangeVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G
    @unpack halosstx,halossty = G

    return SemiLagrangeVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη,
                            halosstx=halosstx,halossty=halossty)
end

###################################################################

"""Preallocate the diagnostic variables and return them as matrices in structs."""
function preallocate(   ::Type{T},
                        ::Type{Tprog},
                        G::Grid) where {T<:AbstractFloat,Tprog<:AbstractFloat}

    RK = RungeKuttaVars{Tprog}(G)
    TD = TendencyVars{Tprog}(G)
    VF = VolumeFluxVars{T}(G)
    VT = VorticityVars{T}(G)
    BN = BernoulliVars{T}(G)
    BD = BottomdragVars{T}(G)
    AH = ArakawaHsuVars{T}(G)
    LP = LaplaceVars{T}(G)
    SM = SmagorinskyVars{T}(G)
    SL = SemiLagrangeVars{T}(G)
    PV = PrognosticVars{T}(G)

    return DiagnosticVars{T,Tprog}(RK,TD,VF,VT,BN,BD,AH,LP,SM,SL,PV)
end

mutable struct PrognosticVars{T<:AbstractFloat}
    u::Array{T,2}           # u-velocity
    v::Array{T,2}           # v-velocity
    η::Array{T,2}           # sea surface height / interface displacement
    sst::Array{T,2}         # tracer / sea surface temperature
end

struct DiagnosticVars{T,Tprog}
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
end

struct ModelSetup{T<:AbstractFloat,Tprog<:AbstractFloat}
    parameters::Parameter
    grid::Grid{T,Tprog}
    constants::Constants{T,Tprog}
    forcing::Forcing{T}
    Prog::PrognosticVars{T}
    Diag::DiagnosticVars{T, Tprog}
end

"""Zero generator function for Grid G as argument."""
function PrognosticVars{T}(G::Grid) where {T<:AbstractFloat}
    @unpack nux,nuy,nvx,nvy,nx,ny = G
    @unpack halo,haloη,halosstx,halossty = G

    u = zeros(T,nux+2halo,nuy+2halo)
    v = zeros(T,nvx+2halo,nvy+2halo)
    η = zeros(T,nx+2haloη,ny+2haloη)
    sst = zeros(T,nx+2halosstx,ny+2halossty)

    return PrognosticVars{T}(u,v,η,sst)
end

function initial_conditions(::Type{T},G::Grid,P::Parameter,C::Constants) where {T<:AbstractFloat}

    ## PROGNOSTIC VARIABLES U,V,η
    @unpack nux,nuy,nvx,nvy,nx,ny = G
    @unpack initial_cond = P
    @unpack Tini = P

    if initial_cond == "rest"

        u = zeros(T,nux,nuy)
        v = zeros(T,nvx,nvy)
        η = zeros(T,nx,ny)

    elseif initial_cond == "ncfile"

        @unpack initpath,init_run_id,init_starti = P
        @unpack init_interpolation = P
        @unpack nx,ny = G

        inirunpath = joinpath(initpath,"run"*@sprintf("%04d",init_run_id))

        # take starti time step from existing netcdf files
        ncstring = joinpath(inirunpath,"u.nc")
        ncu = NetCDF.open(ncstring)

        if init_starti == -1    # replace -1 with length of time dimension
            init_starti = size(ncu.vars["t"])[end]
        end

        u = ncu.vars["u"][:,:,init_starti]

        ncv = NetCDF.open(joinpath(inirunpath,"v.nc"))
        v = ncv.vars["v"][:,:,init_starti]

        ncη = NetCDF.open(joinpath(inirunpath,"eta.nc"))
        η = ncη.vars["eta"][:,:,init_starti]

        # remove singleton time dimension
        u = reshape(u,size(u)[1:2])
        v = reshape(v,size(v)[1:2])
        η = reshape(η,size(η)[1:2])

        nx_old,ny_old = size(η)

        if (nx_old,ny_old) != (nx,ny)
            if init_interpolation           # interpolation in case the grids don't match

                # old grids
                x_T = collect(0.5:nx_old-0.5)
                y_T = collect(0.5:ny_old-0.5)

                # assuming periodic BC for now #TODO make flexible
                x_u = collect(0:nx_old-1)
                y_u = y_T

                x_v = x_T
                y_v = collect(1:ny_old-1)

                # set up interpolation functions
                u_itp = interpolate((x_u,y_u),u,Gridded(Linear()))
                v_itp = interpolate((x_v,y_v),v,Gridded(Linear()))
                η_itp = interpolate((x_T,y_T),η,Gridded(Linear()))

                #TODO in case of interpolation on larger grids
                #TODO make BC adapting to actual BCs used.
                u_etp = extrapolate(u_itp,(Flat(),Flat()))
                v_etp = extrapolate(v_itp,(Flat(),Flat()))
                η_etp = extrapolate(η_itp,(Flat(),Flat()))

                # new grids
                Δx = nx_old/nx
                Δy = ny_old/ny

                x_T_new = collect(Δx/2:Δx:nx_old-Δx/2)
                y_T_new = collect(Δy/2:Δy:ny_old-Δy/2)

                x_u_new = collect(0:Δx:nx_old-Δx)
                y_u_new = y_T_new

                x_v_new = x_T_new
                y_v_new = collect(Δy:Δy:ny_old-Δy)

                # retrieve values and overwrite existing arrays
                u = u_etp(x_u_new,y_u_new)
                v = v_etp(x_v_new,y_v_new)
                η = η_etp(x_T_new,y_T_new)

            else
                throw(error("Grid size $((nx,ny)) doesn't match initial conditions on a $(size(η)) grid."))
            end
        end
    end

    ## SST

    @unpack SSTmin, SSTmax, SSTw, SSTϕ = P
    @unpack SSTwaves_nx,SSTwaves_ny,SSTwaves_p = P
    @unpack sst_initial,scale = P
    @unpack x_T,y_T,Lx,Ly = G

    xx_T,yy_T = meshgrid(x_T,y_T)

    if sst_initial == "south"
        sst = (SSTmin+SSTmax)/2 .+ tanh.(2π*(Ly/(4*SSTw))*(yy_T/Ly .- SSTϕ))*(SSTmin-SSTmax)/2
    elseif sst_initial == "west"
        sst = (SSTmin+SSTmax)/2 .+ tanh.(2π*(Lx/(4*SSTw))*(xx_T/Lx .- SSTϕ))*(SSTmin-SSTmax)/2
    elseif sst_initial == "linear"
        sst = SSTmin .+ yy_T/Ly*(SSTmax-SSTmin)
    elseif sst_initial == "waves"
        sst = waves(xx_T/Lx,yy_T/Ly,SSTwaves_nx,SSTwaves_ny,SSTwaves_p)
    elseif sst_initial == "flat"
        sst = fill(SSTmin,size(xx_T))
    elseif sst_initial == "rect"
        @unpack sst_rect_coords = P
        x0,x1,y0,y1 = sst_rect_coords

        sst = fill(SSTmin,size(xx_T))
        inside = (xx_T/Lx .> x0) .* (xx_T/Lx .< x1) .* (yy_T/Ly .> y0) .* (yy_T/Ly .< y1)
        sst[inside] .= SSTmax
    end

    if initial_cond == "ncfile" && sst_initial == "restart"
        ncsst = NetCDF.open(joinpath(inirunpath,"sst.nc"))
        sst = ncsst.vars["sst"][:,:,init_starti]

        sst = reshape(sst,size(sst)[1:2])
    end

    # Convert to number format T
    # allow for comparable initial conditions via Tini
    sst = T.(Tini.(sst))
    u = T.(Tini.(u))
    v = T.(Tini.(v))
    η = T.(Tini.(η))

    #TODO SST INTERPOLATION
    u,v,η,sst = add_halo(u,v,η,sst,G,P,C)

    return PrognosticVars{T}(u,v,η,sst)
end

function initial_conditions(::Type{T},S::ModelSetup) where {T<:AbstractFloat}

    ## PROGNOSTIC VARIABLES U,V,η
    @unpack nux,nuy,nvx,nvy,nx,ny = S.grid
    @unpack initial_cond = S.parameters
    @unpack Tini = S.parameters

    if initial_cond == "rest"

        u = zeros(T,nux,nuy)
        v = zeros(T,nvx,nvy)
        η = zeros(T,nx,ny)

    elseif initial_cond == "ncfile"

        @unpack initpath,init_run_id,init_starti = S.parameters
        @unpack init_interpolation = S.parameters
        @unpack nx,ny = S.grid

        inirunpath = joinpath(initpath,"run"*@sprintf("%04d",init_run_id))

        # take starti time step from existing netcdf files
        ncstring = joinpath(inirunpath,"u.nc")
        ncu = NetCDF.open(ncstring)

        if init_starti == -1    # replace -1 with length of time dimension
            init_starti = size(ncu.vars["t"])[end]
        end

        u = ncu.vars["u"][:,:,init_starti]

        ncv = NetCDF.open(joinpath(inirunpath,"v.nc"))
        v = ncv.vars["v"][:,:,init_starti]

        ncη = NetCDF.open(joinpath(inirunpath,"eta.nc"))
        η = ncη.vars["eta"][:,:,init_starti]

        # remove singleton time dimension
        u = reshape(u,size(u)[1:2])
        v = reshape(v,size(v)[1:2])
        η = reshape(η,size(η)[1:2])

        nx_old,ny_old = size(η)

        if (nx_old,ny_old) != (nx,ny)
            if init_interpolation           # interpolation in case the grids don't match

                # old grids
                x_T = collect(0.5:nx_old-0.5)
                y_T = collect(0.5:ny_old-0.5)

                # assuming periodic BC for now #TODO make flexible
                x_u = collect(0:nx_old-1)
                y_u = y_T

                x_v = x_T
                y_v = collect(1:ny_old-1)

                # set up interpolation functions
                u_itp = interpolate((x_u,y_u),u,Gridded(Linear()))
                v_itp = interpolate((x_v,y_v),v,Gridded(Linear()))
                η_itp = interpolate((x_T,y_T),η,Gridded(Linear()))

                #TODO in case of interpolation on larger grids
                #TODO make BC adapting to actual BCs used.
                u_etp = extrapolate(u_itp,(Flat(),Flat()))
                v_etp = extrapolate(v_itp,(Flat(),Flat()))
                η_etp = extrapolate(η_itp,(Flat(),Flat()))

                # new grids
                Δx = nx_old/nx
                Δy = ny_old/ny

                x_T_new = collect(Δx/2:Δx:nx_old-Δx/2)
                y_T_new = collect(Δy/2:Δy:ny_old-Δy/2)

                x_u_new = collect(0:Δx:nx_old-Δx)
                y_u_new = y_T_new

                x_v_new = x_T_new
                y_v_new = collect(Δy:Δy:ny_old-Δy)

                # retrieve values and overwrite existing arrays
                u = u_etp(x_u_new,y_u_new)
                v = v_etp(x_v_new,y_v_new)
                η = η_etp(x_T_new,y_T_new)

            else
                throw(error("Grid size $((nx,ny)) doesn't match initial conditions on a $(size(η)) grid."))
            end
        end
    end

    ## SST

    @unpack SSTmin, SSTmax, SSTw, SSTϕ = S.parameters
    @unpack SSTwaves_nx,SSTwaves_ny,SSTwaves_p = S.parameters
    @unpack sst_initial,scale = S.parameters
    @unpack x_T,y_T,Lx,Ly = S.grid

    xx_T,yy_T = meshgrid(x_T,y_T)

    if sst_initial == "south"
        sst = (SSTmin+SSTmax)/2 .+ tanh.(2π*(Ly/(4*SSTw))*(yy_T/Ly .- SSTϕ))*(SSTmin-SSTmax)/2
    elseif sst_initial == "west"
        sst = (SSTmin+SSTmax)/2 .+ tanh.(2π*(Lx/(4*SSTw))*(xx_T/Lx .- SSTϕ))*(SSTmin-SSTmax)/2
    elseif sst_initial == "linear"
        sst = SSTmin .+ yy_T/Ly*(SSTmax-SSTmin)
    elseif sst_initial == "waves"
        sst = waves(xx_T/Lx,yy_T/Ly,SSTwaves_nx,SSTwaves_ny,SSTwaves_p)
    elseif sst_initial == "flat"
        sst = fill(SSTmin,size(xx_T))
    elseif sst_initial == "rect"
        @unpack sst_rect_coords = S.parameters
        x0,x1,y0,y1 = sst_rect_coords

        sst = fill(SSTmin,size(xx_T))
        inside = (xx_T/Lx .> x0) .* (xx_T/Lx .< x1) .* (yy_T/Ly .> y0) .* (yy_T/Ly .< y1)
        sst[inside] .= SSTmax
    end

    if initial_cond == "ncfile" && sst_initial == "restart"
        ncsst = NetCDF.open(joinpath(inirunpath,"sst.nc"))
        sst = ncsst.vars["sst"][:,:,init_starti]

        sst = reshape(sst,size(sst)[1:2])
    end

    # Convert to number format T
    # allow for comparable initial conditions via Tini
    sst = T.(Tini.(sst))
    u = T.(Tini.(u))
    v = T.(Tini.(v))
    η = T.(Tini.(η))

    #TODO SST INTERPOLATION
    u,v,η,sst = add_halo(u,v,η,sst,S)

    return PrognosticVars{T}(u,v,η,sst)
end

"""Create a wave-checkerboard pattern over xx,yy like a nx x ny checkerboard.
p is the power to which the waves are raised. Choose p<1 for rectangles, and
p > 1 for more smootheness."""
function waves(xx::AbstractMatrix,yy::AbstractMatrix,nx::Real,ny::Real,p::Real)
    @boundscheck size(xx) == size(yy) || throw(BoundsError())
    w = sin.(nx*π*xx) .* sin.(ny*π*yy)
    s = sign.(w)
    return s.*abs.(w).^p
end

function time_integration_debug(S::ModelSetup{T,Tprog}) where {T<:AbstractFloat,Tprog<:AbstractFloat}

    Diag = S.Diag
    Prog = S.Prog

    P = S.parameters
    C = S.constants

    @unpack u,v,η,sst = Prog
    # @unpack u0,v0,η0 = Diag.RungeKutta
    # @unpack u1,v1,η1 = Diag.RungeKutta
    # @unpack du,dv,dη = Diag.Tendencies
    # @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
    # @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

    # @unpack um,vm = Diag.SemiLagrange

    # @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
    # @unpack time_scheme,compensated = S.parameters
    # @unpack RKaΔt,RKbΔt = S.constants
    # @unpack Δt_Δ,Δt_Δs = S.constants

    @unpack nt,dtint = S.grid
    # @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid

   
    for i = 1:nt

        # ghost point copy for boundary conditions
        Tcomm = Float32

        # @unpack bc,Tcomm = P

        # @unpack one_minus_α = C
        # one_minus_α = 2
    
        # ghost_points_u_periodic!(Tcomm,u,one_minus_α)
        n,m = size(u)
        @inbounds for j in 1:m
            u[1,j] = T(Tcomm(u[end-3,j]))
            u[2,j] = T(Tcomm(u[end-2,j]))
            u[end-1,j] = T(Tcomm(u[3,j]))
            u[end,j] = T(Tcomm(u[4,j]))
        end

        # ghost_points_v_periodic!(Tcomm,v)
        n,m = size(v)
        @inbounds for j in 1:m
            v[1,j] = T(Tcomm(v[end-3,j]))
            v[2,j] = T(Tcomm(v[end-2,j]))
            v[end-1,j] = T(Tcomm(v[3,j]))
            v[end,j] = T(Tcomm(v[4,j]))
        end

        # ghost_points_η_periodic!(Tcomm,η)
        n,m = size(η)
        @inbounds for j in 1:m
            η[1,j] = T(Tcomm(η[end-1,j]))
            η[end,j] = T(Tcomm(η[2,j]))
        end
        @inbounds for i in 1:n
            η[i,1] = η[i,2]
            η[i,end] = η[i,end-1]
        end

        # copyto!(u1,u)
        # copyto!(v1,v)
        # copyto!(η1,η)

    end

    return nothing 

end

# """ Extends the matrices u,v,η,sst with a halo of ghost points for boundary conditions."""
function add_halo(  u::Array{T,2},
                    v::Array{T,2},
                    η::Array{T,2},
                    sst::Array{T,2},
                    G::Grid,
                    P::Parameter,
                    C::Constants) where {T<:AbstractFloat}

    @unpack nx,ny,nux,nuy,nvx,nvy = G
    @unpack halo,haloη,halosstx,halossty = G

    # Add zeros to satisfy kinematic boundary conditions
    u = cat(zeros(T,halo,nuy),u,zeros(T,halo,nuy),dims=1)
    u = cat(zeros(T,nux+2*halo,halo),u,zeros(T,nux+2*halo,halo),dims=2)

    v = cat(zeros(T,halo,nvy),v,zeros(T,halo,nvy),dims=1)
    v = cat(zeros(T,nvx+2*halo,halo),v,zeros(T,nvx+2*halo,halo),dims=2)

    η = cat(zeros(T,haloη,ny),η,zeros(T,haloη,ny),dims=1)
    η = cat(zeros(T,nx+2*haloη,haloη),η,zeros(T,nx+2*haloη,haloη),dims=2)

    sst = cat(zeros(T,halosstx,ny),sst,zeros(T,halosstx,ny),dims=1)
    sst = cat(zeros(T,nx+2*halosstx,halossty),sst,zeros(T,nx+2*halosstx,halossty),dims=2)

    # SCALING
    @unpack scale,scale_sst = C
    u *= scale
    v *= scale
    sst *= scale_sst

    Tcomm = Float32

    # ghost_points!(u,v,η,P,C)
    # ghost_points_sst!(sst,P,G)

    # ghost_points!(u,v,η,S)

    # ghost_points_u_periodic!(Tcomm,u,one_minus_α)
    n,m = size(u)
    @inbounds for j in 1:m
        u[1,j] = T(Tcomm(u[end-3,j]))
        u[2,j] = T(Tcomm(u[end-2,j]))
        u[end-1,j] = T(Tcomm(u[3,j]))
        u[end,j] = T(Tcomm(u[4,j]))
    end

    # ghost_points_v_periodic!(Tcomm,v)
    n,m = size(v)
    @inbounds for j in 1:m
        v[1,j] = T(Tcomm(v[end-3,j]))
        v[2,j] = T(Tcomm(v[end-2,j]))
        v[end-1,j] = T(Tcomm(v[3,j]))
        v[end,j] = T(Tcomm(v[4,j]))
    end

    # ghost_points_η_periodic!(Tcomm,η)
    n,m = size(η)
    @inbounds for j in 1:m
        η[1,j] = T(Tcomm(η[end-1,j]))
        η[end,j] = T(Tcomm(η[2,j]))
    end
    @inbounds for i in 1:n
        η[i,1] = η[i,2]
        η[i,end] = η[i,end-1]
    end
    n,m = size(sst)

    # corner points are copied twice, but it's faster!
    @inbounds for j in 1:m
        for i ∈ 1:halosstx
            sst[i,j] = T(Tcomm(sst[end-2*halosstx+i,j]))
            sst[end-halosstx+i,j] = T(Tcomm(sst[halosstx+i,j]))
        end
    end

    @inbounds for j ∈ 1:halossty
        for i in 1:n
            sst[i,j] = sst[i,halossty+1]
            sst[i,end-j+1] = sst[i,end-halossty]
        end
    end

    return u,v,η,sst
end


##### Original versions ######################################################################################################################

# """ Extends the matrices u,v,η,sst with a halo of ghost points for boundary conditions."""
function add_halo(  u::Array{T,2},
                    v::Array{T,2},
                    η::Array{T,2},
                    sst::Array{T,2},
                    S::ModelSetup) where {T<:AbstractFloat}

    @unpack nx,ny,nux,nuy,nvx,nvy = S.grid
    @unpack halo,haloη,halosstx,halossty = S.grid

    # Add zeros to satisfy kinematic boundary conditions
    u = cat(zeros(T,halo,nuy),u,zeros(T,halo,nuy),dims=1)
    u = cat(zeros(T,nux+2*halo,halo),u,zeros(T,nux+2*halo,halo),dims=2)

    v = cat(zeros(T,halo,nvy),v,zeros(T,halo,nvy),dims=1)
    v = cat(zeros(T,nvx+2*halo,halo),v,zeros(T,nvx+2*halo,halo),dims=2)

    η = cat(zeros(T,haloη,ny),η,zeros(T,haloη,ny),dims=1)
    η = cat(zeros(T,nx+2*haloη,haloη),η,zeros(T,nx+2*haloη,haloη),dims=2)

    sst = cat(zeros(T,halosstx,ny),sst,zeros(T,halosstx,ny),dims=1)
    sst = cat(zeros(T,nx+2*halosstx,halossty),sst,zeros(T,nx+2*halosstx,halossty),dims=2)

    # SCALING
    @unpack scale,scale_sst = S.constants
    u *= scale
    v *= scale
    sst *= scale_sst

    Tcomm = Float32

    # ghost_points!(u,v,η,S)

    # ghost_points_u_periodic!(Tcomm,u,one_minus_α)
    n,m = size(u)
    @inbounds for j in 1:m
        u[1,j] = T(Tcomm(u[end-3,j]))
        u[2,j] = T(Tcomm(u[end-2,j]))
        u[end-1,j] = T(Tcomm(u[3,j]))
        u[end,j] = T(Tcomm(u[4,j]))
    end

    # ghost_points_v_periodic!(Tcomm,v)
    n,m = size(v)
    @inbounds for j in 1:m
        v[1,j] = T(Tcomm(v[end-3,j]))
        v[2,j] = T(Tcomm(v[end-2,j]))
        v[end-1,j] = T(Tcomm(v[3,j]))
        v[end,j] = T(Tcomm(v[4,j]))
    end

    # ghost_points_η_periodic!(Tcomm,η)
    n,m = size(η)
    @inbounds for j in 1:m
        η[1,j] = T(Tcomm(η[end-1,j]))
        η[end,j] = T(Tcomm(η[2,j]))
    end
    @inbounds for i in 1:n
        η[i,1] = η[i,2]
        η[i,end] = η[i,end-1]
    end
    n,m = size(sst)

    # corner points are copied twice, but it's faster!
    @inbounds for j in 1:m
        for i ∈ 1:halosstx
            sst[i,j] = T(Tcomm(sst[end-2*halosstx+i,j]))
            sst[end-halosstx+i,j] = T(Tcomm(sst[halosstx+i,j]))
        end
    end

    @inbounds for j ∈ 1:halossty
        for i in 1:n
            sst[i,j] = sst[i,halossty+1]
            sst[i,end-j+1] = sst[i,end-halossty]
        end
    end
    
    return u,v,η,sst

end


function run_enzyme(::Type{T}=Float32;     # number format
    kwargs...                             # all additional parameters
    ) where {T<:AbstractFloat}

    P = Parameter(T=T;kwargs...)
    return run_enzyme(T,P)
end

function run_enzyme(P::Parameter)
    @unpack T = P
    return run_enzyme(T,P)
end

function run_enzyme(::Type{T},P::Parameter) where {T<:AbstractFloat}

    Tprog = Float32

    G = Grid{T,Tprog}(P)
    C = Constants{T,Tprog}(P,G)
    F = Forcing{T}(P,G)

    Prog = initial_conditions(Tprog,G,P,C)
    Diag = preallocate(T,Tprog,G)

    # one structure with everything inside 
    S = ModelSetup{T,Tprog}(P,G,C,F,Prog,Diag)
    dS = deepcopy(S)

    autodiff(Reverse, time_integration_debug, Duplicated(S, dS))

    return nothing 

end

run_enzyme(nx=5, Ndays=1)
