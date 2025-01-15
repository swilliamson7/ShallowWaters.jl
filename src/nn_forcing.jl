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

function NN_momentum(u, v, S)

    Diag = S.Diag

    @unpack γ₀, zb_filtered, N  = S.parameters

    @unpack nqx, nqy = Diag.NNVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.NNVars
    @unpack ζ, D, Dhat = Diag.NNVars
    @unpack ζT, DT, ζDhat = Diag.NNVars
    @unpack weights_offdiagonal, weights_diagonal = Diag.NNVars
    @unpack diagonal_outdim, diagonal_indim, offdiagonal_indim, offdiagonal_outdim = Diag.NNVars

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

    # Stretch deformation, cell centers (with halo)
    @inbounds for j ∈ 1:nTh
        for k ∈ 1:mTh
            Dhat[k,j] = dudx[k,j+1] - dvdy[k+1,j]
        end
    end

    # Here we define the models to be used for the forcing, currently both just a single layer
    # We'll have two models, one will produce the off diagonal term in T (T_xy = T_yx = ζD̃), and
    # the second will produce the diagonal terms (T_xx = ζ^2 - ζ D, T_yy = ζ^2 + ζ D)
    # The weights are defined as inputs here so that we can tune them

    # the output of diagonal_model will (for first efforts) be a 2*nx*ny (cell centers)
    # array, which will then be split and reshaped into two nx by ny arrays, the first of
    # which is ζ^2 - ζD ("interpolated" to cell centers), and ζ^2 + ζD (also "interpolated")
    # to cell centers). We want the output on cell centers so that when we take the gradient
    # of the whole tensor T, everything livess where we need it to and we can easily find
    # S_u and S_v
    diagonal_model = Lux.Dense(diagonal_indim => diagonal_outdim, relu;init_weight=weights_diagonal)

    # the output of off_diagonalmodel will be a nqy*nqy array, which we can reshape into
    # a nqx by nqy (cell corners) array, intended to be ζD̃, the offdiagonal term in T
    offdiagonal_model = Lux.Dense(offdiagonal_indim => offdiagonal_outdim, relu;init_weight=weights_offdiagonal)

    # now we want to feed ζ, D, and D̃ to the NN
    temp1 = diagonal_model([reshape(ζ,nqx*nqy); reshape(D,nqx*nqy)])
    temp2 = offdiagonal_model([reshape(ζ,nqx*nqy); reshape(Dhat,mTh*nTh)])

    # reshape the outputs
    ζT = reshape(temp1[1:nx*ny], nx, ny)
    DT = reshape(temp1[nx*ny+1:end], nx, ny)
    ζDhat = reshape(temp2, nqx, nqy)

    # for j ∈ 2:nq-1
    #     for k ∈ 2:mq-1

    #         temp1 = diagonal_model([reshape(ζ[j-1:j+1,k-1:k+1], 9); reshape(D[j-1:j+1,k-1:k+1], 9)])

    #         S_u[j,k] = temp1[1]
    #         S_v[j,k] = temp1[2]

    #     end
    # end

    # for j ∈ 1:nuy
    #     for k ∈ 1:nvx

    #         for j ∈ 2:nTh-1
    #             for k ∈ 2:mTh-1

    #                 temp2 = offdiagonal_model([reshape(ζ[j-1:j+1,k-1:k+1], 9); reshape(Dhat[j-1:j+1,k-1:k+1], 9)])

    #             end
    #         end

    #         S_u[j,k] += temp2[1]
    #         S_v[j,k] += temp2[2]

    #     end
    # end

end

# writing a "function that acts like a neural net", i.e. will take in 
# u, v, and the model S, and output the elements of the parameterization
# all without using Lux.jl
function handwritten_NN_momentum(u, v, S)

    Diag = S.Diag

    @unpack γ₀, zb_filtered, N  = S.parameters

    @unpack nqx, nqy = Diag.NNVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.NNVars
    @unpack ζ, D, Dhat = Diag.NNVars
    @unpack ζT, DT, ζDhat = Diag.NNVars
    @unpack weights_offdiagonal, weights_diagonal = Diag.NNVars
    @unpack diagonal_outdim, diagonal_indim, offdiagonal_indim, offdiagonal_outdim = Diag.NNVars

    @unpack S_u, S_v = Diag.ZBVars

    @unpack Δ, scale, f₀ = S.grid
    @unpack halo, haloη, ep, nux, nuy, nvx, nvy = S.grid

    mTh,nTh = size(Dhat)

    κ_BT = - γ₀ * Δ^2

    ShallowWaters.∂x!(dudx, S.Prog.u)
    ShallowWaters.∂y!(dudy, S.Prog.u)

    ShallowWaters.∂x!(dvdx, S.Prog.v)
    ShallowWaters.∂y!(dvdy, S.Prog.v)

    # Relative vorticity and shear deformation, cell corners
    @inbounds for j ∈ 1:nqx
        for k ∈ 1:nqy
            ζ[k,j] = dvdx[k+1,j+1] - dudy[k+1,j+1]
            D[k,j] = dudy[k+1,j+1] + dvdx[k+1,j+1]
        end
    end

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

    for j ∈ 2:nqx-1
        for k ∈ 2:nqy-1

            temp11, temp22 = hw_diagonal_model(reshape(ζ[j-1:j+1,k-1:k+1], 9),
                reshape(D[j-1:j+1,k-1:k+1], 9),
                reshape(Dhat[j:j+1,k:k+1], 4),
                weights_diagonal
            )

        end
    end

    # for j ∈ 1:nuy
    #     for k ∈ 1:nvx

    for j ∈ 2:nTh-1
        for k ∈ 2:mTh-1

            temp12 = hw_offdiagonal_model(reshape(ζ[j:j+1,k:k+1], 4),
                reshape(D[j:j+1,k:k+1], 4),
                reshape(Dhat[j-1:j+1,k-1:k+1], 9),
                weights_offdiagonal
            )

        end
    end

    #     end
    # end

end

# This function takes in ζ, D, and Dhat, applies the "neural net" 
# (here just a combination of linear operators and a subsequent nonlinear
# operator, tbd). We have two: one for the diagonal terms and one for the
# off-diagonal terms
function hw_diagonal_model(ζ, D, Dhat, weights)

    linear = weights * [ζ; D; Dhat]
    out = relu.(linear)

    return out[1], out[2]

end

function hw_offdiagonal_model(ζ, D, Dhat, weights)

    linear = weights * [ζ; D; Dhat]
    out = relu(linear)

    return out

end