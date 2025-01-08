"""
This function will add an additional forcing term S, meant to encapsulate the
eddy forcing generated at subgrid-scales. However, here we utilize a neural net to do so,
rather than a deterministic function.

The function takes 
    u: the x-direction velocity
    v: the y-direction velocity
    S: the weight matrix to be used within the neural net
The function should return the elements of the forcing tensor S,
    T_xx,
    T_xy,
    T_yx,
and
    T_yy
The inputs to the NN will be the vorticity, shear, and stretch deformation fields,
as is done in the Zanna & Bolton parameterization
"""

function NN_momentum(u, v, S)

    Diag = S.Diag

    @unpack γ₀, zb_filtered, N  = S.parameters

    @unpack nqx, nqy = Diag.NNVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.NNVars
    @unpack ζ, D, Dhat = Diag.NNVars
    @unpack ζT, DT, ζDhat = Diag.NNVars
    @unpack weights_offdiagonal, weights_diagonal = Diag.NNVars

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

    # First computing the fields we will feed to the NN

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
    diagonal_model = Lux.Dense(weights_diagonal, false, elu)

    # the output of off_diagonalmodel will be a nqy*nqy array, which we can reshape into
    # a nqx by nqy (cell corners) array, intended to be ζD̃, the offdiagonal term in T
    offdiagonal_model = Lux.Dense(weights_offdiagonal, false, elu)

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