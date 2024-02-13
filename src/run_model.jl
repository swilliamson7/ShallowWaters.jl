"""

    u,v,η,sst = run_model()

runs ShallowWaters with default parameters as defined in src/DefaultParameters.jl

# Examples
```jldoc
julia> u,v,η,sst = run_model(Float64,nx=200,output=true)
```
"""

## Forward run modified ###########################################################################

function run_model(::Type{T}=Float32;     # number format
    kwargs...                             # all additional parameters
    ) where {T<:AbstractFloat}

    P = Parameter(T=T;kwargs...)
    return run_model(T,P)
end

function run_model(P::Parameter)
    @unpack T = P
    return run_model(T,P)
end

function run_model(::Type{T},P::Parameter) where {T<:AbstractFloat}

    @unpack Tprog = P

    G = Grid{T,Tprog}(P)
    C = Constants{T,Tprog}(P,G)
    F = Forcing{T}(P,G)
    # S = ModelSetup{T,Tprog}(P,G,C,F)

    Prog = initial_conditions(Tprog,G,P,C)
    Diag = preallocate(T,Tprog,G)

    # one structure with everything already inside
    S = ModelSetup{T,Tprog}(P,G,C,F,Prog,Diag,0)
    S, e, P = time_integration_withreturn(S)

    return S, e, P

end

#### for checking that the modified time_integration still works ##################################

function run_check(::Type{T}=Float32;     # number format
    kwargs...                             # all additional parameters
    ) where {T<:AbstractFloat}

    P = Parameter(T=T;kwargs...)
    return run_check(T,P)
end

function run_check(P::Parameter)
    @unpack T = P
    return run_check(T,P)
end

function run_check(::Type{T},P::Parameter) where {T<:AbstractFloat}

    @unpack Tprog = P

    G = Grid{T,Tprog}(P)
    C = Constants{T,Tprog}(P,G)
    F = Forcing{T}(P,G)
    # S = ModelSetup{T,Tprog}(P,G,C,F)

    Prog = initial_conditions(Tprog,G,P,C)
    Diag = preallocate(T,Tprog,G)

    # one structure with everything already inside
    S = ModelSetup{T,Tprog}(P,G,C,F,Prog,Diag,0)
    P = time_integration_mine(S)

    return P

end

##################################################################

## Built in Enzyme run #######################################################################

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

    @unpack Tprog = P

    G = Grid{T,Tprog}(P)
    C = Constants{T,Tprog}(P,G)
    F = Forcing{T}(P,G)

    Prog = initial_conditions(Tprog,G,P,C)
    Diag = preallocate(T,Tprog,G)

    # one structure with everything already inside
    S = ModelSetup{T,Tprog}(P,G,C,F,Prog,Diag,0)
    dS = deepcopy(S)
    autodiff(Reverse, ShallowWaters.time_integration_nofeedback, Duplicated(S, dS))

    return S, dS

end

#### Setup run #########################################################################################################

# This just returns the initial structure S

function run_setup(::Type{T}=Float32;     # number format
    kwargs...                             # all additional parameters
    ) where {T<:AbstractFloat}

    P = Parameter(T=T;kwargs...)
    return run_setup(T,P)
end

function run_setup(P::Parameter)
    @unpack T = P
    return run_setup(T,P)
end

function run_setup(::Type{T},P::Parameter) where {T<:AbstractFloat}

    @unpack Tprog = P

    G = Grid{T,Tprog}(P)
    C = Constants{T,Tprog}(P,G)
    F = Forcing{T}(P,G)
    # S = ModelSetup{T,Tprog}(P,G,C,F)

    Prog = initial_conditions(Tprog,G,P,C)
    Diag = preallocate(T,Tprog,G)

    # one structure with everything inside
    S = ModelSetup{T,Tprog}(P,G,C,F,Prog,Diag,0)

    return S

end

#### Original ##################################################################################

# In order to run the following you need to change lines in model_setup.jl

function mk_run_model(::Type{T}=Float32;     # number format
    kwargs...                            # all additional parameters
    ) where {T<:AbstractFloat}

    P = Parameter(T=T;kwargs...)
    return mk_run_model(T,P)
end

function mk_run_model(P::Parameter)
    @unpack T = P
    return mk_run_model(T,P)
end

function mk_run_model(::Type{T},P::Parameter) where {T<:AbstractFloat}

    @unpack Tprog = P

    G = Grid{T,Tprog}(P)
    C = Constants{T,Tprog}(P,G)
    F = Forcing{T}(P,G)
    S = ModelSetup{T,Tprog}(P,G,C,F,0)

    Prog = initial_conditions(Tprog,S)
    Diag = preallocate(T,Tprog,G)

    Prog = time_integration(Prog,Diag,S)
    return Prog

end

