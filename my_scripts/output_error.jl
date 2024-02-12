include("../src/ShallowWaters.jl")
using .ShallowWaters 
using NetCDF, Printf, Dates, Interpolations
using Enzyme#main

Enzyme.API.runtimeActivity!(true)

### Checkpointing check

function checkpoint_function(S)

    # setup 
    Diag = S.Diag
    Prog = S.Prog

    i = S.parameters.i

    # these are the specific lines that seem to trigger the Enzyme error
    # feedback, output initialisation and storing initial conditions
    feedback = feedback_init(S)
    netCDFfiles = ShallowWaters.NcFiles(feedback,S)
    ShallowWaters.output_nc!(0,netCDFfiles,Prog,Diag,S)

    nans_detected = false
    t = 0                       # model time

    for S.parameters.i = 1:nt

        # feedback and output
        feedback.i = i
        feedback!(Prog,feedback,S)
        ShallowWaters.output_nc!(S.parameters.i,netCDFfiles,Prog,Diag,S)       # uses u0,v0,η0

        # adding this causes a "break outside loop" error and the code just won't run
        # if feedback.nans_detected
        #     break
        # end

    end

    return nothing

end

# working (fingers crossed)
function run_checkpointing()


    S = ShallowWaters.run_setup(nx = 128,
    Ndays = 1,
    zb_forcing_momentum=false,
    zb_filtered=false,
    output=false
    )

    dS = Enzyme.Compiler.make_zero(Core.Typeof(S), IdDict(), S)

    autodiff(Enzyme.ReverseWithPrimal, checkpoint_function, Duplicated(S, dS))

    return S, dS

end

@time S365_energy, dS365_energy = run_checkpointing()