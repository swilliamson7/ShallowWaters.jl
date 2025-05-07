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

# as of 05/06/25 there are dimension issues here when L_ratio \neq 1

using Lux, Random

function NN_momentum(u, v, S)

    Diag = S.Diag

    @unpack zb_filtered, N = S.parameters

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

    ζ = cat(zeros(1,nqy),ζ,zeros(1,nqy),dims=1)
    ζ = cat(zeros(nqx+2,1),ζ,zeros(nqx+2,1),dims=2)
    D = cat(zeros(1,nqy),D,zeros(1,nqy),dims=1)
    D = cat(zeros(nqx+2,1),D,zeros(nqx+2,1),dims=2)

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

    corner_layers = Lux.Dense(corner_indim => corner_outdim, relu)
    center_layers = Lux.Dense(center_indim => center_outdim, relu)

    ps_corner, st_corner = Lux.setup(Random.default_rng(), corner_layers)
    ps_center, st_center = Lux.setup(Random.default_rng(), center_layers)

    corner_params = (weight=weights_corner, bias=ps_corner.bias)
    center_params = (weight=weights_center, bias=ps_center.bias)

    corner_model = StatefulLuxLayer{true}(corner_layers, corner_params, st_corner)
    center_model = StatefulLuxLayer{true}(center_layers, center_params, st_center)

    for j ∈ 1:nqx
        for k ∈ 1:nqy

            temp11, temp22 = corner_model([reshape(ζ[j:j+2,k:k+2], 9);
                reshape(D[j:j+2,k:k+2], 9);
                reshape(Dhat[j:j+1,k:k+1], 4)]
            )

            T11[j,k] = temp11
            T22[j,k] = temp22

        end
    end

    for j ∈ 2:mTh-1
        for k ∈ 2:nTh-1

            temp12 = center_model([reshape(ζ[j:j+1,k:k+1], 4);
                reshape(D[j:j+1,k:k+1], 4);
                reshape(Dhat[j-1:j+1,k-1:k+1], 9)]
            )

            T12[j-1,k-1] = temp12[1]

        end
    end

end

# writing a "function that acts like a neural net", i.e. will take in 
# u, v, and the model S, and output the elements of the parameterization
# all without using Lux.jl
function handwritten_NN_momentum(u, v, S)

    Diag = S.Diag

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

    ζ = cat(zeros(1,nqy),ζ,zeros(1,nqy),dims=1)
    ζ = cat(zeros(nqx+2,1),ζ,zeros(nqx+2,1),dims=2)
    D = cat(zeros(1,nqy),D,zeros(1,nqy),dims=1)
    D = cat(zeros(nqx+2,1),D,zeros(nqx+2,1),dims=2)

    # Stretch deformation, cell centers (with halo)
    @inbounds for j ∈ 1:nTh
        for k ∈ 1:mTh
            Dhat[k,j] = dudx[k,j+1] - dvdy[k+1,j]
        end
    end

    # we have two functions defined below, intended to mimic the inner operations of the 
    # NN, for now this will just be a single layer
    # I'm still sticking with two neurals, one to determine the off-diagonal term
    # and the other to determine the diagonal terms, T_12 and T_11, T_22, respectively

    for j ∈ 1:nqx
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

    for j ∈ 2:mTh-1
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
    out = relu(linear)

    return out

end