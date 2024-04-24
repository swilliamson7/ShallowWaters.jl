function run_checkpoint_model(::Type{T}=Float32;     # number format
    kwargs...                             # all additional parameters
    ) where {T<:AbstractFloat}

    P = ShallowWaters.Parameter(T=T;kwargs...)
    return run_checkpoint_model(T,P)
end

function run_checkpoint_model(P::Parameter)
    @unpack T = P
    return run_checkpoint_model(T,P)
end

function run_checkpoint_model(::Type{T},P::Parameter) where {T<:AbstractFloat}

    @unpack Tprog = P

    G = ShallowWaters.Grid{T,Tprog}(P)
    C = ShallowWaters.Constants{T,Tprog}(P,G)
    F = ShallowWaters.Forcing{T}(P,G)

    Prog = ShallowWaters.initial_conditions(Tprog,G,P,C)
    Diag = ShallowWaters.preallocate(T,Tprog,G)

    # one structure with everything already inside 
    S = ShallowWaters.ModelSetup{T,Tprog}(P,G,C,F,Prog,Diag,0)
    Prog = ShallowWaters.checkpointed_integration_integration(S)

    return Prog

end