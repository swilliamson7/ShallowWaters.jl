using NetCDF, Parameters, Printf, Dates, Interpolations
using JLD2, Plots, LaTeXStrings

using Enzyme#main
Enzyme.API.maxtypeoffset!(3500)

include("../src/default_parameters.jl")
include("../src/grid.jl")
include("../src/constants.jl")
include("../src/forcing.jl")
include("../src/preallocate.jl")
include("../src/model_setup.jl")
include("../src/initial_conditions.jl")

include("../src/time_integration.jl")
include("../src/ghost_points.jl")
include("../src/rhs.jl")
include("../src/gradients.jl")
include("../src/interpolations.jl")
include("../src/PV_advection.jl")
include("../src/continuity.jl")
include("../src/bottom_drag.jl")
include("../src/diffusion.jl")
include("../src/tracer_advection.jl")

include("../src/feedback.jl")
include("../src/output.jl")
include("../src/run_model.jl")

include("../src/zanna_bolton_forcing.jl")

function for_video(S::ModelSetup{T,Tprog}) where {T<:AbstractFloat,Tprog<:AbstractFloat}

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
    thickness!(Diag.VolumeFluxes.h,η,S.forcing.H)
    Ix!(Diag.VolumeFluxes.h_u,Diag.VolumeFluxes.h)
    Iy!(Diag.VolumeFluxes.h_v,Diag.VolumeFluxes.h)
    Ixy!(Diag.Vorticity.h_q,Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)
    advection_coriolis!(urhs,vrhs,ηrhs,Diag,S)
    PVadvection!(Diag,S)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)

    # feedback, output initialisation and storing initial conditions
    feedback = feedback_init(S)
    netCDFfiles = NcFiles(feedback,S)
    output_nc!(0,netCDFfiles,Prog,Diag,S)

    nans_detected = false
    t = 0           # model time
    energy_anim = Animation()
    energy = zeros(nt)
    for i = 1:nt

        # ghost point copy for boundary conditions
        ghost_points!(u,v,η,S)
        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        if compensated
            fill!(du_sum,zero(Tprog))
            fill!(dv_sum,zero(Tprog))
            fill!(dη_sum,zero(Tprog))
        end

        for rki = 1:RKo
            if rki > 1
                ghost_points!(u1,v1,η1,S)
            end

            # type conversion for mixed precision
            u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
            v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
            η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

            rhs!(u1rhs,v1rhs,η1rhs,Diag,S,t)          # momentum only
            continuity!(u1rhs,v1rhs,η1rhs,Diag,S,t)   # continuity equation 

            if rki < RKo
                caxb!(u1,u,RKbΔt[rki],du)   #u1 .= u .+ RKb[rki]*Δt*du
                caxb!(v1,v,RKbΔt[rki],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
                caxb!(η1,η,RKbΔt[rki],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
            end

            if compensated      # accumulate tendencies
                axb!(du_sum,RKaΔt[rki],du)   
                axb!(dv_sum,RKaΔt[rki],dv)
                axb!(dη_sum,RKaΔt[rki],dη)
            else    # sum RK-substeps on the go
                axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη
            end
        end

        if compensated
            # add compensation term to total tendency
            axb!(du_sum,-1,du_comp)             
            axb!(dv_sum,-1,dv_comp)
            axb!(dη_sum,-1,dη_comp)

            axb!(u0,1,du_sum)   # update prognostic variable with total tendency
            axb!(v0,1,dv_sum)
            axb!(η0,1,dη_sum)
            
            dambmc!(du_comp,u0,u,du_sum)    # compute new compensation
            dambmc!(dv_comp,v0,v,dv_sum)
            dambmc!(dη_comp,η0,η,dη_sum)
        end

        ghost_points!(u0,v0,η0,S)

        # type conversion for mixed precision
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        η0rhs = convert(Diag.PrognosticVarsRHS.η,η0)

        # ADVECTION and CORIOLIS TERMS
        # although included in the tendency of every RK substep,
        # only update every nstep_advcor steps if nstep_advcor > 0
        if dynamics == "nonlinear" && nstep_advcor > 0 && (i % nstep_advcor) == 0
            UVfluxes!(u0rhs,v0rhs,η0rhs,Diag,S)
            advection_coriolis!(u0rhs,v0rhs,η0rhs,Diag,S)
        end

        # DIFFUSIVE TERMS - SEMI-IMPLICIT EULER
        # use u0 = u^(n+1) to evaluate tendencies, add to u0 = u^n + rhs
        # evaluate only every nstep_diff time steps
        if (i % nstep_diff) == 0
            bottom_drag!(u0rhs,v0rhs,η0rhs,Diag,S)
            diffusion!(u0rhs,v0rhs,Diag,S)
            add_drag_diff_tendencies!(u0,v0,Diag,S)
            ghost_points_uv!(u0,v0,S)
        end

        t += dtint

        # TRACER ADVECTION
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)  # copy back as add_drag_diff_tendencies changed u0,v0
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        tracer!(i,u0rhs,v0rhs,Prog,Diag,S)

        # feedback and output
        feedback.i = i
        feedback!(Prog,feedback,S)
        output_nc!(i,netCDFfiles,Prog,Diag,S)       # uses u0,v0,η0

        if feedback.nans_detected
            break
        end

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

        # for animation 
        if i in 1:225:S.grid.nt
            temp = PrognosticVars{Tprog}(remove_halo(u,v,η,sst,S)...)
            energy = [zeros(1,S.parameters.nx); (temp.u.^2)] + [zeros(S.parameters.nx,1) (temp.v.^2)] / (S.grid.nx * S.grid.ny) 
            frame(energy_anim, heatmap(energy', title="E($i), temp" , 
                xlabel=L"x", ylabel=L"y", c=:dense, dpi=300, clim=(0, 2)))
        end

        # compute and store energy 
        # temp = PrognosticVars{Tprog}(remove_halo(u,v,η,sst,S)...)
        # energy[i] = (sum(Diag.Vorticity.KEu) + sum(Diag.Vorticity.KEv)) / (S.grid.nx * S.grid.ny) 

    end

    # finalise feedback and output
    feedback_end!(feedback)
    output_close!(netCDFfiles,feedback,S)

    gif(energy_anim, "energy_integration_temp_90days_092623_fps10.gif", fps = 10)

    if S.parameters.return_time
        return feedback.tend - feedback.t0
    else
        # return energy, PrognosticVars{Tprog}(remove_halo(u,v,η,sst,S)...)
        # return S.Prog, energy
        return nothing
    end


end

# hourly_saves = 1:225:S.grid.nt

# initial conditions 
init_cond_lr = load_object("../data_files/128_spinup_wforcing/spinup_states_128_withzb_10years_101023.jld2")
# init_cond_hr = load_object("../data_files/flat_domain/spinup_states_512_10years.jld2")

# lr_energy = load_object("../data_files/flat_domain/energy_during_spinup_128_10years.jld2")
# hr_energy = load_object("../data_files/flat_domain/energy_during_spinup_512_10years.jld2")

# without extra forcing 
S = run_setup(nx=128,Ndays=90,zb_forcing=true)
u,v,η,sst = add_halo(init_cond_lr.u, init_cond_lr.v, init_cond_lr.η, init_cond_lr.sst, S.grid, S.parameters, S.constants)

S.Prog.u .= u 
S.Prog.v .= v 
S.Prog.η .= η 
S.Prog.sst .= sst

for_video(S)

P = remove_halo(S.Prog.u, S.Prog.v, S.Prog.η, S.Prog.sst, S.grid, S.constants)

# energy_without, P_without = for_video(S)

# with extra forcing 
# S_new = run_setup(nx=128,Ndays=365,zb_forcing=true)

# S_new.Prog.u .= init_cond_lr.u 
# S_new.Prog.v .= init_cond_lr.v 
# S_new.Prog.η .= init_cond_lr.η 
# S_new.Prog.sst .= init_cond_lr.sst

# energy_with, P_with = for_video(S_new)

# plot(lr_energy, xlabel="x", ylabel="y", label="Without extra forcing", dpi=300) 
# plot!(energy_with, label="With extra forcing") 
# plot!(hr_energy, label="High resolution, no extra forcing")

# savefig("for_meeting.png")