"""
This function will add an additional forcing term S, meant to encapsulate the
eddy forcing generated at subgrid-scales. However, here we utilize a neural net to do so,
rather than a deterministic function.

The inputs (as of 09/05/24) are
    u: the x-direction velocity
    v: the y-direction velocity
    W: the weight matrix to be used within the neural net
The function should return the elements of the forcing tensor S,
    T_xx,
    T_xy,
    T_yx,
and
    T_yy
The inputs we receive will be the vorticity, shear, and stretch deformation fields,
and the outputs will be 
"""

function NN_momentum(u, v, S)

    Diag = S.Diag

    @unpack γ₀, zb_filtered, N  = S.parameters

    @unpack nqx, nqy = Diag.NNVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.NNVars
    @unpack ζ, D, Dhat = Diag.NNVars
    @unpack ζT, DT = Diag.NNVars
    @unpack weights_center, weights_corner = Diag.NNVars

    @unpack S_u, S_v = Diag.ZBVars

    @unpack Δ, scale, f₀ = S.grid
    @unpack halo, haloη, ep, nux, nuy, nvx, nvy = S.grid

    κ_BC = - γ₀ * Δ^2

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
    # We have one model for the diagnostic values on the cell corners, and one for the diagnostic
    # quantity on the cell centers
    corner_model = Dense(weights_corner, false, relu)
    center_model = Dense(weights_center, false, relu)

    for j ∈ 2:nq-1
        for k ∈ 2:mq-1

            temp1 = corner_model([reshape(ζ[j-1:j+1,k-1:k+1], 9); reshape(D[j-1:j+1,k-1:k+1], 9)])

            S_u[j,k] = temp1[1]
            S_v[j,k] = temp1[2]

        end
    end

    for j ∈ 1:nuy
        for k ∈ 1:nvx

            for j ∈ 2:nTh-1
                for k ∈ 2:mTh-1

                    temp2 = center_model([reshape(Dhat[j-1:j+1,k-1:k+1], 9)])

                end
            end

            S_u[j,k] += temp2[1]
            S_v[j,k] += temp2[2]

        end
    end

end