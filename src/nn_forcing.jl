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

function NN_momentum_corners(u, v, S, Diag, weights)

    @unpack γ₀, zb_filtered, N  = S.parameters
    @unpack nqx, nqy = Diag.NNVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.NNVars
    @unpack ζ, D = Diag.NNVars

    @unpack ζT, DT = Diag.NNVars

    @unpack S_u, S_v = Diag.ZBVars
    @unpack Δ, scale, f₀ = S.grid

    @unpack halo, haloη, ep, nux, nuy, nvx, nvy = S.grid

    κ_BC = - γ₀ * Δ^2

    ∂x!(dudx, u)
    ∂y!(dudy, u)

    ∂x!(dvdx, v)
    ∂y!(dvdy, v)

    T_xx = zeros(127,128)
    T_xy = zeros(127,128)
    T_yy = zeros(128,127)
    T_yx = zeros(128,127)

    # Relative vorticity and shear deformation, cell corners
    @inbounds for j ∈ 1:nq
        for k ∈ 1:mq
            ζ[k,j] = dvdx[k+1,j+1] - dudy[k+1,j+1]
            D[k,j] = dudy[k+1,j+1] + dvdx[k+1,j+1]
        end
    end

    corner_model = Dense(weights, relu)

    for j ∈ 2:nq-1
        for k ∈ 2:mq-1

            temp = corner_model([reshape(ζ[j-1:j+1,k-1:k+1], 9); reshape(D[j-1:j+1,k-1:k+1], 9)])
            

        end
    end

    return S_u

end

function NN_momentum_centers(u, v, S, Diag, weights)

    @unpack γ₀, zb_filtered, N  = S.parameters
    @unpack ζ, ζsq, D, Dsq, Dhat, Dhatsq, Dhatq = Diag.ZBVars
    @unpack ζD, ζDT, ζDhat, ζsqT, trace = Diag.ZBVars
    @unpack ζpDT = Diag.ZBVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.ZBVars

    @unpack dζDdx, dζDhatdy, dtracedx = Diag.ZBVars
    @unpack dζDhatdx, dζDdy, dtracedy = Diag.ZBVars
    @unpack S_u, S_v = Diag.ZBVars
    @unpack Δ, scale, f₀ = S.grid

    @unpack G = Diag.ZBVars
    @unpack ζD_filtered, ζDhat_filtered, trace_filtered = Diag.ZBVars

    @unpack halo, haloη, ep, nux, nuy, nvx, nvy = S.grid

    κ_BC = - γ₀ * Δ^2

    ∂x!(dudx, u)
    ∂y!(dudy, u)

    ∂x!(dvdx, v)
    ∂y!(dvdy, v)



end

function NN_momentum(u, v, S, Diag, weights1, weights2)



end