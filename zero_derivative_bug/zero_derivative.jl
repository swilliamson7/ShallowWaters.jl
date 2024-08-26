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

            ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)          # momentum only
            ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)   # continuity equation

            if rki < RKo
                ShallowWaters.caxb!(u1,u,RKbΔt[rki],du)   #u1 .= u .+ RKb[rki]*Δt*du
                ShallowWaters.caxb!(v1,v,RKbΔt[rki],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
                ShallowWaters.caxb!(η1,η,RKbΔt[rki],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
            end

            if compensated      # accumulate tendencies
                ShallowWaters.axb!(du_sum,RKaΔt[rki],du)
                ShallowWaters.axb!(dv_sum,RKaΔt[rki],dv)
                ShallowWaters.axb!(dη_sum,RKaΔt[rki],dη)
            else    # sum RK-substeps on the go
                ShallowWaters.axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                ShallowWaters.axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                ShallowWaters.axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη
            end
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


function run_script(S, Ndays)

    # 225 steps = 1 day of integration in the 128 model
    data_steps = 1:1:88
    nx=10

    S_outer = deepcopy(S)

    dS = Enzyme.Compiler.make_zero(Core.Typeof(S), IdDict(), S)
    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false
    )

    autodiff(Enzyme.ReverseWithPrimal, checkpointed_integration, Active, Duplicated(S, dS), Const(revolve))

    # fn = sprint() do io
    #     Enzyme.Compiler.enzyme_code_llvm(io, checkpointed_integration, Const, Tuple{Duplicated{typeof(S)}, Const{typeof(revolve)}}; dump_module=true)
    # end

    # @code_warntype checkpointed_integration(S,revolve)

     # enzyme_deriv = dS.parameters.γ₀
     enzyme_deriv = dS.Prog.u[5, 5]

     steps = [50, 40, 30, 20, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

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

         S_inner.Prog.u[5, 5] += s

         J_inner = checkpointed_integration(S_inner, revolve)

         push!(diffs, (J_inner - J_outer) / s)

     end

    return S, dS, enzyme_deriv, diffs

end

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
nx=10,
Ndays = Ndays,
zb_forcing_dissipation=true,
γ₀ = 0.3,
data_steps=data_steps)

S, dS, enzyme_deriv, diffs = run_script(S,Ndays)

# diffs, enzyme_deriv = check_derivative(dS, 1, [0.0], 225:225:225*(Ndays-1))