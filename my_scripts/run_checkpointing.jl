include("../src/ShallowWaters.jl")
using .ShallowWaters 
using NetCDF, Parameters, Printf, Dates, Interpolations
using JLD2
using Enzyme#main
using Checkpointing

Enzyme.API.runtimeActivity!(true)

### Checkpointing check

function checkpoint_function(S, scheme)

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

    ###### for optimization problem
    # data_steps = S.parameters.data_steps
    # data = S.parameters.data
    # J = S.parameters.J
    i = S.parameters.i
    # j = S.parameters.j
    #######

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

    # adding causes bug in Enzyme, don't really need these functions
    # (or at least it doesn't seem like I do)
    # feedback, output initialisation and storing initial conditions
    # feedback = feedback_init(S)
    # netCDFfiles = ShallowWaters.NcFiles(feedback,S)
    # ShallowWaters.output_nc!(0,netCDFfiles,Prog,Diag,S)

    # nans_detected = false
    t = 0                       # model time
    # run integration loop with checkpointing
    @checkpoint_struct scheme S for S.parameters.i = 1:nt

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(u,v,η,S)
        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        if time_scheme == "RK"   # classic RK4,3 or 2

            if compensated
                fill!(du_sum,zero(Tprog))
                fill!(dv_sum,zero(Tprog))
                fill!(dη_sum,zero(Tprog))
            end

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

            if compensated
                # add compensation term to total tendency
                ShallowWaters.axb!(du_sum,-1,du_comp)             
                ShallowWaters.axb!(dv_sum,-1,dv_comp)
                ShallowWaters.axb!(dη_sum,-1,dη_comp)

                ShallowWaters.axb!(u0,1,du_sum)   # update prognostic variable with total tendency
                ShallowWaters.axb!(v0,1,dv_sum)
                ShallowWaters.axb!(η0,1,dη_sum)
                
                ShallowWaters.dambmc!(du_comp,u0,u,du_sum)    # compute new compensation
                ShallowWaters.dambmc!(dv_comp,v0,v,dv_sum)
                ShallowWaters.dambmc!(dη_comp,η0,η,dη_sum)
            end

        elseif time_scheme == "SSPRK2"  # s-stage 2nd order SSPRK

            for rki = 1:RKs
                if rki > 1
                    ShallowWaters.ghost_points_η!(η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)        # momentum only

                # the update step
                ShallowWaters.axb!(u1,Δt_Δs,du)       # u1 = u1 + Δt/(s-1)*RHS(u1)
                ShallowWaters.axb!(v1,Δt_Δs,dv)

                # semi-implicit for continuity equation, use new u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)
                ShallowWaters.axb!(η1,Δt_Δs,dη)       # η1 = η1 + Δt/(s-1)*RHS(u1)
            end

            a = 1/RKs
            b = (RKs-1)/RKs
            ShallowWaters.cxayb!(u0,a,u,b,u1)
            ShallowWaters.cxayb!(v0,a,v,b,v1)
            ShallowWaters.cxayb!(η0,a,η,b,η1)
        
        elseif time_scheme == "SSPRK3"  # s-stage 3rd order SSPRK

            @unpack s,kn,mn,kna,knb,Δt_Δnc,Δt_Δn = S.constants.SSPRK3c

            # if compensated
            #     fill!(du_sum,zero(Tprog))
            #     fill!(dv_sum,zero(Tprog))
            #     fill!(dη_sum,zero(Tprog))
            # end

            for rki = 2:s+1       # number of stages (from 2:s+1 to match Ketcheson et al 2014)
                if rki > 2
                    ShallowWaters.ghost_points_η!(η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                if rki == kn    # special case combining more previous stages  
                    ShallowWaters.dxaybzc!(u1,kna,u1,knb,u0,Δt_Δnc,du)
                    ShallowWaters.dxaybzc!(v1,kna,v1,knb,v0,Δt_Δnc,dv)
                else                                # normal update case
                    ShallowWaters.axb!(u1,Δt_Δn,du)   
                    ShallowWaters.axb!(v1,Δt_Δn,dv)

                    # if compensated
                    #     axb!(du_sum,Δt_Δn,du)   
                    #     axb!(dv_sum,Δt_Δn,dv)
                    # end
                end

                # semi-implicit for continuity equation, use new u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                if rki == kn
                    ShallowWaters.dxaybzc!(η1,kna,η1,knb,η0,Δt_Δnc,dη)
                else
                    ShallowWaters.axb!(η1,Δt_Δn,dη)
                    # if compensated
                    #     axb!(dη_sum,Δt_Δn,dη)
                    # end
                end

                # special stage that is needed later for the kn-th stage, store in u0,v0,η0 therefore
                # or for the last step, as u0,v0,η0 is used as the last step's result of any RK scheme.
                if rki == mn || rki == s+1
                    copyto!(u0,u1)
                    copyto!(v0,v1)
                    ShallowWaters.ghost_points_η!(η1,S)
                    copyto!(η0,η1)
                end
            end
            
        elseif time_scheme == "4SSPRK3"   # 4-stage SSPRK3
        
            for rki = 1:4
                if rki > 1
                    ShallowWaters.ghost_points!(u1,v1,η1,S)
                end

                # type conversion for mixed precision
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

                ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)

                ShallowWaters.caxb!(u0,u1,Δt_Δ,du)        # store Euler update into u0,v0
                ShallowWaters.caxb!(v0,v1,Δt_Δ,dv)
                ShallowWaters.cxab!(u1,1/2,u1,u0)         # average u0,u1 and store in u1
                ShallowWaters.cxab!(v1,1/2,v1,v0)         # same

                # semi-implicit for continuity equation, use u1,v1 to calcualte dη
                ShallowWaters.ghost_points_uv!(u1,v1,S)
                u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
                v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
                ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)
                
                ShallowWaters.caxb!(η0,η1,Δt_Δ,dη)    # store Euler update into η0
                ShallowWaters.cxab!(η1,1/2,η1,η0)         # average η0,η1 and store in η1

                if rki == 3
                    ShallowWaters.cxayb!(u1,2/3,u,1/3,u1)
                    ShallowWaters.cxayb!(v1,2/3,v,1/3,v1)
                    ShallowWaters.cxayb!(η1,2/3,η,1/3,η1)
                elseif rki == 4
                    copyto!(u0,u1)
                    copyto!(v0,v1)
                    copyto!(η0,η1)
                end
            end
        end

        ShallowWaters.ghost_points!(u0,v0,η0,S)

        # type conversion for mixed precision
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        η0rhs = convert(Diag.PrognosticVarsRHS.η,η0)

        # ADVECTION and CORIOLIS TERMS
        # although included in the tendency of every RK substep,
        # only update every nstep_advcor steps if nstep_advcor > 0
        if dynamics == "nonlinear" && nstep_advcor > 0 && (i % nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs,v0rhs,η0rhs,Diag,S)
            ShallowWaters.advection_coriolis!(u0rhs,v0rhs,η0rhs,Diag,S)
        end

        # DIFFUSIVE TERMS - SEMI-IMPLICIT EULER
        # use u0 = u^(n+1) to evaluate tendencies, add to u0 = u^n + rhs
        # evaluate only every nstep_diff time steps
        if (S.parameters.i % nstep_diff) == 0
            ShallowWaters.bottom_drag!(u0rhs,v0rhs,η0rhs,Diag,S)
            ShallowWaters.diffusion!(u0rhs,v0rhs,Diag,S)
            ShallowWaters.add_drag_diff_tendencies!(u0,v0,Diag,S)
            ShallowWaters.ghost_points_uv!(u0,v0,S)
        end

        t += dtint

        # TRACER ADVECTION
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)  # copy back as add_drag_diff_tendencies changed u0,v0
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        ShallowWaters.tracer!(i,u0rhs,v0rhs,Prog,Diag,S)

        # # feedback and output
        # feedback.i = i
        # feedback!(Prog,feedback,S)
        # ShallowWaters.output_nc!(S.parameters.i,netCDFfiles,Prog,Diag,S)       # uses u0,v0,η0

        # if feedback.nans_detected
        #     break
        # end

        #### cost function evaluation

        # if S.parameters.i in S.parameters.data_steps

        #     temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(u,v,η,sst,S)...)
        #     energy_lr = (sum(temp.u.^2) + sum(temp.v.^2)) / (S.grid.nx * S.grid.ny)

        #     # spacially averaged energy objective function
        #     S.parameters.J += (energy_lr - S.parameters.data[S.parameters.j])^2

        #     S.parameters.j += 1

        # end

        #############################################################

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

    end

    temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(u,v,η,sst,S)...)
    return temp.η[24,24]

    # temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(u,v,η,sst,S)...)
    # S.parameters.J = (sum(temp.u.^2) + sum(temp.v.^2)) / (S.grid.nx * S.grid.ny)
    # return S.parameters.J

end

function check_derivative(dS)

    # u = S.Prog.u
    # v = S.Prog.v
    # η = S.Prog.η
    # sst = S.Prog.sst

    du = dS.Prog.u
    dv = dS.Prog.v
    dη = dS.Prog.η
    dsst = dS.Prog.sst

    # P = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(u,v,η,sst,S)...)
    # dP = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(du,dv,dη,dsst,dS)...)

    # du = dP.u
    # dv = dP.v
    # dη = dP.η

    enzyme_calculated_derivative = dv[24, 24]

    steps = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]

    S_outer = ShallowWaters.run_setup(nx = 50,
    Ndays = 1,
    zb_forcing_momentum=false,
    zb_filtered=false,
    output=false
    )

    S_unperturbed, _, _ = ShallowWaters.time_integration_withreturn(S_outer)

    diffs = []

    for s in  steps

        S_inner = ShallowWaters.run_setup(nx = 50,
        Ndays = 1,
        zb_forcing_momentum=false,
        zb_filtered=false,
        output=false
        )

        S_inner.Prog.v[24, 24] += s

        S_perturbed, _, _ = ShallowWaters.time_integration_withreturn(S_inner)

        push!(diffs, (S_unperturbed.Prog.u[24, 24] - S_perturbed.Prog.u[24, 24]) / s)

    end

    return diffs, enzyme_calculated_derivative

end

# working (fingers crossed)
function run_checkpointing()


    S = ShallowWaters.run_setup(nx = 128,
    Ndays = 365,
    zb_forcing_momentum=false,
    zb_filtered=false,
    # initial_cond = "ncfile",
    # initpath="./data_files_gamma0.3/128_spinup_noforcing",
    output=false
    )

    dS = Enzyme.Compiler.make_zero(Core.Typeof(S), IdDict(), S)
    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt, snaps; verbose=1, gc=true, write_checkpoints=false)

    autodiff(Enzyme.ReverseWithPrimal, checkpoint_function, Duplicated(S, dS), revolve)

    return S, dS

end

# not working
function run_energy_checkpointing()

    energy_high_resolution = load_object("data_files_gamma0.3/512_post_spinup_4years/energy_post_spinup_512_4years_012524.jld2")
    grid_scale = 4

    # aiming to have data about every 30 days
    data_steps = 6733:6733
    data = [energy_high_resolution[6733*grid_scale]]

    S = ShallowWaters.run_setup(nx = 128,
        Ndays = 30,
        zb_forcing_dissipation=true,
        zb_filtered=true,
        data_steps=data_steps,
        data=data,
        initial_cond="ncfile",
        initpath="./data_files_gamma0.3/128_spinup_wforcing_dissipation_wfilter_1pass/"
    )

    dS = Enzyme.Compiler.make_zero(Core.Typeof(S), IdDict(), S)

    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false
    )

    autodiff(Enzyme.ReverseWithPrimal,
        checkpoint_function,
        Duplicated(S, dS),
        revolve
    )

    return S, dS

end

@time S365_energy, dS365_energy = run_checkpointing()