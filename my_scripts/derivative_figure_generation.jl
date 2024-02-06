include("../src/ShallowWaters.jl")
using .ShallowWaters
using NetCDF, Parameters, Printf, Dates, Interpolations
using JLD2
using Enzyme#main
using Checkpointing

Enzyme.API.runtimeActivity!(true)

# You need to change the return value of checkpointed_time_integration if different derivatives are 
# desired

function run_checkpointing()

    S = ShallowWaters.run_setup(nx = 50,
    Ndays = 1,
    zb_forcing_momentum=false,
    zb_filtered=false,
    output=false
    )

    dS = Enzyme.Compiler.make_zero(Core.Typeof(S),
        IdDict(),
        S
    )

    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false
    )

    autodiff(Reverse,
        ShallowWaters.checkpointed_time_integration,
        Duplicated(S, dS),
        revolve
    )

    u = S.Prog.u
    v = S.Prog.v
    η = S.Prog.η
    sst = S.Prog.sst

    du = dS.Prog.u
    dv = dS.Prog.v
    dη = dS.Prog.η
    dsst = dS.Prog.sst

    P = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(u,v,η,sst,S)...)
    dP = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(du,dv,dη,dsst,dS)...)

    return P, dP, dS

end

# gradient check to make sure I'm not being a silly scientist
function check_derivative(dP)

    du = dP.u
    dv = dP.v
    dη = dP.η

    enzyme_calculated_derivative = du[24, 24]

    steps = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]

    S_outer = ShallowWaters.run_setup(nx = 50,
    Ndays = 1,
    zb_forcing_momentum=false,
    zb_filtered=false,
    output=false
    )

    _, _, P_unperturbed = ShallowWaters.time_integration_withreturn(S_outer)

    diffs = []

    for s in  steps

        S_inner = ShallowWaters.run_setup(nx = 50,
        Ndays = 1,
        zb_forcing_momentum=false,
        zb_filtered=false,
        output=false
        )

        S_inner.Prog.u[24, 24] += s

        _, _, P_inner_unperturbed = ShallowWaters.time_integration_withreturn(S_inner)

        push!(diffs, (P_unperturbed.η[25, 25] - P_inner_unperturbed.η[25, 25]) / s)

    end

    return diffs, enzyme_calculated_derivative

end


_, dP, _ = run_checkpointing()

diffs, enzyme_calculated_derivatve = check_derivative(dP)