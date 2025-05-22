"""
Both of these functions will add an additional forcing term S, meant to encapsulate the
eddy forcing generated at subgrid-scales. However, here we utilize a neural net to do so,
rather than a deterministic function.

The function takes 
    u: the x-direction velocity
    v: the y-direction velocity
    S: the weight matrix to be used within the neural net
The function should return the elements of the forcing tensor S,
    T_11,
    T_12,
    T_12,
and
    T_22
The inputs to the NN will be the vorticity, shear, and stretch deformation fields,
as is done in the Zanna & Bolton parameterization. We note that, as in the 
Zanna and Bolton parameterization, for initial test runs we assume T_12 \equiv T_21
"""

# !!!!!! as of 05/06/25 there are dimension issues here when L_ratio \neq 1, so don't use on non-square domains

using Lux, Random

function NN_momentum(u, v, S)

    Diag = S.Diag

    T = Float32 
    @unpack zb_filtered, N = S.parameters

    @unpack nqx, nqy = Diag.NNVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.NNVars
    @unpack γ₀, ζ, D, Dhat = Diag.NNVars
    @unpack ζT, DT, ζDhat = Diag.NNVars
    @unpack T11, T22, T12, T21 = Diag.NNVars
    @unpack model_corner, model_center = Diag.NNVars
    @unpack center_layers, corner_layers = Diag.NNVars
    @unpack compiled_corner, compiled_center = Diag.NNVars
    @unpack corner_outdim, corner_indim, center_indim, center_outdim = Diag.NNVars

    @unpack S_u, S_v = Diag.ZBVars

    @unpack Δ, scale, f₀ = S.grid
    @unpack halo, haloη, ep, nux, nuy, nvx, nvy = S.grid

    mq,nq = size(ζ)
    mTh,nTh = size(Dhat)

    κ_BT = - γ₀ * Δ^2

    ∂x!(dudx, u)
    ∂y!(dudy, u)

    ∂x!(dvdx, v)
    ∂y!(dvdy, v)

    # Relative vorticity and shear deformation, cell corners
    @inbounds for j ∈ 1:nq
        for k ∈ 1:mq
            ζ[k,j] = dvdx[k+1,j+1] - dudy[k+1,j+1]
            D[k,j] = dudy[k+1,j+1] + dvdx[k+1,j+1]
        end
    end

    ζ = cat(zeros(T, 1,nqy),ζ,zeros(T, 1,nqy),dims=1)
    ζ = cat(zeros(T, nqx+2,1),ζ,zeros(T, nqx+2,1),dims=2)
    D = cat(zeros(T, 1,nqy),D,zeros(T, 1,nqy),dims=1)
    D = cat(zeros(T, nqx+2,1),D,zeros(T, nqx+2,1),dims=2)

    # Stretch deformation, cell centers (with halo)
    @inbounds for j ∈ 1:nTh
        for k ∈ 1:mTh
            Dhat[k,j] = dudx[k,j+1] - dvdy[k+1,j]
        end
    end

    # Here we define the models to be used for the forcing, currently both just a single layer
    # We'll have two models, one will produce the off diagonal term in T (T_12 = T_21), and
    # the second will produce the diagonal terms (T_11, T_22)
    # The weights are defined as inputs to the NN so that we can tune them with adjoint based optimization

    corner_input = Array{T}(undef, 9+9+4, nqx, nqy)

    @inbounds for j ∈ 1:nqx
        for k ∈ 1:nqy
            Base.copyto!(@view(corner_input[1:9, j, k]), ζ[j:j+2,k:k+2])
            Base.copyto!(@view(corner_input[10:18, j, k]), D[j:j+2,k:k+2])
            Base.copyto!(@view(corner_input[19:22, j, k]), Dhat[j:j+1,k:k+1])
        end
    end

    @static if false
        @inbounds for j ∈ 1:nqx
            for k ∈ 1:nqy

                Base.copyto!(@view(corner_input[1:9]), ζ[j:j+2,k:k+2])
                Base.copyto!(@view(corner_input[10:18]), D[j:j+2,k:k+2])
                Base.copyto!(@view(corner_input[19:22]), Dhat[j:j+1,k:k+1])

                temp11, temp22 = if compiled_corner isa Nothing
                    Lux.apply(corner_layers, corner_input, model_corner[1], model_corner[2])[1]
                else
                    compiled_corner(corner_layers, Reactant.to_rarray(corner_input), model_corner[1], model_corner[2])[1]
                end

                T11[j,k] = temp11
                T22[j,k] = temp22

            end
        end
    else

        result = if compiled_corner isa Nothing
            Lux.apply(corner_layers, corner_input, model_corner[1], model_corner[2])[1]
        else
            compiled_corner(corner_layers, Reactant.to_rarray(corner_input), model_corner[1], model_corner[2])[1]
        end

        T11 .= result[1, :, :]
        T22 .= result[2, :, :]

    end

    center_input = Array{T}(undef, 4+4+9, mTh-2,nTh-2)
    @show  "run", size(center_input)
    @inbounds for j ∈ 2:mTh-1
        for k ∈ 2:nTh-1
            Base.copyto!(@view(center_input[1:4, j, k]), ζ[j:j+1,k:k+1])
            Base.copyto!(@view(center_input[5:8, j, k]), D[j:j+1,k:k+1])
            Base.copyto!(@view(center_input[9:17, j, k]), Dhat[j-1:j+1,k-1:k+1])
        end
    end

    @static if false
        @inbounds for j ∈ 2:mTh-1
            for k ∈ 2:nTh-1

                Base.copyto!(@view(center_input[1:4, j, k]), ζ[j:j+1,k:k+1])
                Base.copyto!(@view(center_input[5:8, j, k]), D[j:j+1,k:k+1])
                Base.copyto!(@view(center_input[9:17, j, k]), Dhat[j-1:j+1,k-1:k+1])

                temp12 = if compiled_center isa Nothing
                    Lux.apply(center_layers, center_input, model_center[1], model_center[2])[1]
                else
                    compiled_center(center_layers, Reactant.to_rarray(center_input), model_center[1], model_center[2])[1]
                end

                T12[j-1,k-1] = temp12

            end
        end
    else

        result = if compiled_center isa Nothing
            Lux.apply(center_layers, center_input, model_center[1], model_center[2])[1]
        else
            compiled_center(center_layers, Reactant.to_rarray(center_input), model_center[1], model_center[2])[1]
        end

        T12 .= result[1, :, :]

    end

end

# writing a "function that acts like a neural net", i.e. will take in 
# u, v, and the model S, and output the elements of the parameterization
# all without using Lux.jl
function handwritten_NN_momentum(u, v, S)

    Diag = S.Diag

    T = Float32

    @unpack zb_filtered, N  = S.parameters

    @unpack nqx, nqy = Diag.NNVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.NNVars
    @unpack γ₀, ζ, D, Dhat = Diag.NNVars
    @unpack ζT, DT, ζDhat = Diag.NNVars
    @unpack T11, T22, T12, T21 = Diag.NNVars
    @unpack weights_center, weights_corner = Diag.NNVars
    @unpack corner_outdim, corner_indim, center_indim, center_outdim = Diag.NNVars

    @unpack S_u, S_v = Diag.ZBVars

    @unpack Δ, scale, f₀ = S.grid
    @unpack halo, haloη, ep, nux, nuy, nvx, nvy = S.grid

    mTh,nTh = size(Dhat)

    κ_BT = - γ₀ * Δ^2

    ShallowWaters.∂x!(dudx, u)
    ShallowWaters.∂y!(dudy, u)

    ShallowWaters.∂x!(dvdx, v)
    ShallowWaters.∂y!(dvdy, v)

    # Relative vorticity and shear deformation, cell corners
    @inbounds for j ∈ 1:nqx
        for k ∈ 1:nqy
            ζ[k,j] = dvdx[k+1,j+1] - dudy[k+1,j+1]
            D[k,j] = dudy[k+1,j+1] + dvdx[k+1,j+1]
        end
    end

    ζ = cat(zeros(T,1,nqy),ζ,zeros(T,1,nqy),dims=1)
    ζ = cat(zeros(T,nqx+2,1),ζ,zeros(T,nqx+2,1),dims=2)
    D = cat(zeros(T,1,nqy),D,zeros(T,1,nqy),dims=1)
    D = cat(zeros(T,nqx+2,1),D,zeros(T,nqx+2,1),dims=2)

    # Stretch deformation, cell centers (with halo)
    @inbounds for j ∈ 1:nTh
        for k ∈ 1:mTh
            Dhat[k,j] = dudx[k,j+1] - dvdy[k+1,j]
        end
    end

    # we have two functions defined below, intended to mimic the inner operations of the 
    # NN, for now this will just be a single hidden layer
    # I'm still sticking with two neurals, one to determine the off-diagonal term
    # and the other to determine the diagonal terms, T_12 and T_11, T_22, respectively

    @inbounds for j ∈ 1:nqx
        for k ∈ 1:nqy

            temp11, temp22 = hw_corner_model(reshape(ζ[j:j+2,k:k+2], 9),
                reshape(D[j:j+2,k:k+2], 9),
                reshape(Dhat[j:j+1,k:k+1], 4),
                weights_corner
            )

            T11[j,k] = temp11
            T22[j,k] = temp22

        end
    end

    @inbounds for j ∈ 2:mTh-1
        for k ∈ 2:nTh-1

            temp12 = hw_center_model(reshape(ζ[j:j+1,k:k+1], 4),
                reshape(D[j:j+1,k:k+1], 4),
                reshape(Dhat[j-1:j+1,k-1:k+1], 9),
                weights_center
            )

            T12[j-1,k-1] = temp12[1]

        end
    end


end

# This function takes in ζ, D, and Dhat, applies the "neural net" 
# (here just a combination of linear operators and a subsequent nonlinear
# operator, tbd). We have two: one for the diagonal terms that live on cell corners
# and one for the off-diagonal terms that live in cell centers
function hw_corner_model(ζ, D, Dhat, weights)

    linear = weights * [ζ; D; Dhat]
    out = relu.(linear)

    return out[1], out[2]

end

function hw_center_model(ζ, D, Dhat, weights)

    linear = weights * [ζ; D; Dhat]
    out = relu.(linear)

    return out

end