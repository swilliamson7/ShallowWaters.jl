# Assuming the information we have is solely a spacially averaged kinetic energy

include("../src/ShallowWaters.jl")
using .ShallowWaters
using Enzyme#main
using Checkpointing
using JLD2

# Enzyme.API.runtimeActivity!(true)

function run_energy_checkpointing()

    energy_high_resolution = load_object("data_files_gamma0.3/512_post_spinup_4years/energy_post_spinup_512_4years_012524.jld2")
    grid_scale = 4

    # aiming to have data about every 30 days
    data_steps = 6750*12
    data = energy_high_resolution[6750*12*grid_scale]

    S = ShallowWaters.run_setup(nx = 128,
        Ndays = 365,
        zb_forcing_dissipation=true,
        zb_filtered=true,
        data_steps=data_steps,
        data=data,
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
        ShallowWaters.checkpointed_time_integration,
        Duplicated(S, dS),
        revolve
    )

    return S, dS

end