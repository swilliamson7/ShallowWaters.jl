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
using NetCDF

Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.runtimeActivity!(true)

using Parameters
using Optim

function checkpointed_integration(S, scheme)

    # run integration loop with checkpointing
    loop(S, scheme)

    return S.parameters.J

end

function loop(S,scheme)

    # eta_avg = zeros(128,128)
    eta_avg = 0.0

    @checkpoint_struct scheme S for S.parameters.i = 1:S.grid.nt
    # for S.parameters.i = 1:S.grid.nt

        # Cost function evaluation

        if S.parameters.i in S.parameters.data_steps

            temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S.Prog.u,
            S.Prog.v,
            S.Prog.η,
            S.Prog.sst,S)...)

            eta_avg = eta_avg + temp.η[50,50]

            S.parameters.J += sum((eta_avg / S.parameters.i))
            S.parameters.j += 1

        else

            temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S.Prog.u,
            S.Prog.v,
            S.Prog.η,
            S.Prog.sst,S)...)

            eta_avg = eta_avg + temp.η[50, 50]

        end

    end

    return nothing

end

function llvm_error(Ndays)


    # 225 steps = 1 day of integration of the low res model
    # 1800 steps = 1 day of integration of the high res model

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
        nx=128,
        Ndays = Ndays,
        zb_forcing_dissipation=true,
        γ₀ = 0.3
    )

    dS = Enzyme.Compiler.make_zero(Core.Typeof(S), IdDict(), S)
    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false
    )

    autodiff(Enzyme.ReverseWithPrimal, Const(checkpointed_integration), Duplicated(S, dS), Const(revolve))

    return S, dS

end

S, dS = llvm_error(1)