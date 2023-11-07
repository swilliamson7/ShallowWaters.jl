"""
This function will compute an additional forcing term that will (hopefully) act
as the eddy parameterization. I'm doing this by 
(1) adding a new structure that will preallocate all of the operators I need (called ZB_momentum)
(2) I compute the main operators that appear in the new term: ξ, D, Dhat. ξ is the relative vorticity,
D is the shear deformation of the flow field (both of these live on cell corners), and Dhat is the stretch
deformation of the flow field (this lives on the cell centers)
(3) Using ξ, D, and Dhat I then compute the individual pieces of the forcing that appear, comments
below explain on which grid they live and so on 
The first few functions are in regards to the convolutional kernel that's supposed to help the 
parameterization behave better over time
"""
@inline function Gconvolve(array)
    return (array[1, 1] + 2*array[1, 2] + array[1, 3] + 2*array[2, 1] + 4*array[2, 2] + 2*array[2, 3] + array[3, 1] + 2*array[3, 2] + array[3, 3]) / 16
end
@inline function Gconvolve12all(array)
    return (array[1, 1] + 2*array[1, 2] + array[1, 3] + 2*array[2, 1] + 4*array[2, 2] + 2*array[2, 3]) / 16
end
@inline function Gconvolve23all(array)
    return (2*array[1, 1] + 4*array[1, 2] + 2*array[1, 3] + array[2, 1] + 2*array[2, 2] + array[2, 3]) / 16
end
@inline function Gconvolveall12(array)
    return (array[1, 1] + 2*array[1, 2] + 2*array[2, 1] + 4*array[2, 2] + array[3, 1] + 2*array[3, 2]) / 16
end
@inline function Gconvolveall23(array)
    return (2*array[1, 1] + array[1, 2] + 4*array[2, 1] + 2*array[2, 2] + 2*array[3, 1] + array[3, 2]) / 16
end

function ZB_momentum(u, v, S, Diag) 

    @unpack γ₀, zb_filtered, N  = S.parameters
    @unpack ξ, ξsq, D, Dsq, Dhat, Dhatsq, Dhatq = Diag.ZBVars
    @unpack ξD, ξDT, ξDhat, ξsqT, trace = Diag.ZBVars
    @unpack ξpDT = Diag.ZBVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.ZBVars

    @unpack dξDdx, dξDhatdy, dtracedx = Diag.ZBVars
    @unpack dξDhatdx, dξDdy, dtracedy = Diag.ZBVars
    @unpack S_u, S_v = Diag.ZBVars
    @unpack Δ, scale, f₀ = S.grid

    @unpack G = Diag.ZBVars
    @unpack ξD_filtered, ξDhat_filtered, trace_filtered = Diag.ZBVars

    @unpack halo, haloη, ep, nux, nuy, nvx, nvy = S.grid

    κ_BC = - γ₀ * Δ^2

    ∂x!(dudx, u)
    ∂y!(dudy, u)

    ∂x!(dvdx, v)
    ∂y!(dvdy, v)

    mq,nq = size(ξ)
    mTh,nTh = size(Dhat)
    mT,nT = size(trace)

    ##### check #########
    @boundscheck (mq+2,nq+2) == size(dvdx) || throw(BoundsError())
    @boundscheck (mTh+1,nTh+1) == size(dvdx) || throw(BoundsError())
    @boundscheck (mq+2+ep,nq+2) == size(dudy) || throw(BoundsError())
    
    # Relative vorticity and shear deformation, cell corners
    for j ∈ 1:nq
        for k ∈ 1:mq
            ξ[k,j] = dvdx[k+1,j+1] - dudy[k+1,j+1]
            D[k,j] = dudy[k+1,j+1] + dvdx[k+1,j+1]
        end
    end

    # Stretch deformation, cell centers (with halo)
    for j ∈ 1:nTh
        for k ∈ 1:mTh
            Dhat[k,j] = dudx[k,j+1] - dvdy[k+1,j]
        end
    end
    
    ξsq .= κ_BC .* ξ.^2
    Dsq .= D.^2
    Dhatsq .= Dhat.^2

    # Trace computation (second term in forcing term), only keeping ξ^2 and no other terms 
    # Ixy!(ξpDT, ξsq + Dsq)
    Ixy!(ξsqT, ξsq)

    # Computing ξ ⋅ D and placing on cell centers 
    ξD .= κ_BC .* (ξ .* D)
    Ixy!(ξDT, ξD)

    # Computing ξ ⋅ Dhat, cell corners 
    Ixy!(Dhatq, Dhat)
    @inbounds for j ∈ 1:nq
        for k ∈ 1:mq 
            ξDhat[k,j] = κ_BC * ξ[k,j] * Dhatq[k,j]
        end
    end
    # for kj in eachindex(ξDhat,ξ,Dhatq) 
    #     # ξDhat[kj] = κ_BC * ξ[kj] * Dhatq[kj]
    #     ξDhat[kj] = ξ[kj] * Dhatq[kj]
    # end

    # Now, before computing the final divergence, we want to apply a
    # filter to each element of S a total of N times 

    if zb_filtered 
        # applying the filter to ξsqT, doing this in three stages:
        # (1) applying the filter only to the internal points
        # (2) applying the filter to points on the boundary, just ignoring
        # the portions of G that don't have a corresponding point in the grid
        # (3) applying the filter to points on the corner, still just ignoring
        # the portions of G that don't have a corresponding point in the grid
        # G12all = view(G, 1:2, :)
        # G23all = view(G, 2:3, :)
        # Gall12 = view(G, :, 1:2) 
        # Gall23 = view(G, :, 2:3)
        # G1212 = view(G, 1:2, 1:2)
        # G1223 = view(G, 1:2, 2:3)
        # G2312 = view(G, 2:3, 1:2)
        # G2323 = view(G, 2:3, 2:3)
        for i = 1:N
            for j ∈ 2:nT-1 
                for k ∈ 2:mT-1 
                    trace_filtered[k,j] = Gconvolve(@view ξsqT[k-1:k+1,j-1:j+1])
                    trace_filtered[k,j] = Gconvolve(@view ξDT[k-1:k+1,j-1:j+1])
                end
            end
            for h ∈ 2:nT-1
                trace_filtered[1,h] = Gconvolve12all(@view ξsqT[1:2,h-1:h+1])
                trace_filtered[mT,h] = Gconvolve23all(@view ξsqT[mT-1:mT,h-1:h+1])
                ξD_filtered[1,h] = Gconvolve12all(@view ξDT[1:2,h-1:h+1])
                ξD_filtered[mT,h] = Gconvolve23all(@view ξDT[mT-1:mT,h-1:h+1])
                # trace_filtered[1,h] = sum(G12all .* ξsqT[1,h])
                # trace_filtered[mT,h] = sum(G23all .* ξsqT[mT,h])
                # ξD_filtered[1,h] = sum(G12all .* ξDT[1,h])
                # ξD_filtered[mT,h] = sum(G23all .* ξDT[mT,h])
            end
            for v ∈ 2:mT-1 
                trace_filtered[v,1] = Gconvolveall12(ξsqT[v-1:v+1,1:2])
                trace_filtered[v,nT] = Gconvolveall23(ξsqT[v-1:v+1,nT-1:nT])
                ξD_filtered[v,1] = Gconvolveall12(ξDT[v-1:v+1,1:2])
                ξD_filtered[v,nT] = Gconvolveall23(ξDT[v-1:v+1,nT-1:nT])
                # trace_filtered[v,1] = sum(Gall12 .* ξsqT[v,1])
                # trace_filtered[v,nT] = sum(Gall23 .* ξsqT[v,nT])
                # ξD_filtered[v,1] = sum(Gall12 .* ξDT[v,1])
                # ξD_filtered[v,nT] = sum(Gall23 .* ξDT[v,nT])
            end 

            trace_filtered[1,1] = (ξsqT[1,1] + 2*ξsqT[1,2] + 2*ξsqT[2,1] + 4*ξsqT[2,2]) 
            trace_filtered[1,nT] = (2*ξsqT[1,nT-1] + ξsqT[1,nT] + 4*ξsqT[2,nT-1] + 2*ξsqT[2,nT]) 
            trace_filtered[mT,1] = (2*ξsqT[mT-1,1] + 4*ξsqT[mT-1,2] + ξsqT[mT,1] + 2*ξsqT[mT,2]) 
            trace_filtered[mT,nT] = (4*ξsqT[mT-1,nT-1] + 2*ξsqT[mT-1,nT] + 2*ξsqT[mT,nT-1] + ξsqT[mT,nT]) 

            ξD_filtered[1,1] = (ξDT[1,1] + 2*ξDT[1,2] + 2*ξDT[2,1] + 4*ξDT[2,2])
            ξD_filtered[1,nT] = (2*ξDT[1,nT-1] + ξDT[1,nT] + 4*ξDT[2,nT-1] + 2*ξDT[2,nT]) 
            ξD_filtered[mT,1] = (2*ξDT[mT-1,1] + 4*ξDT[mT-1,2] + ξDT[mT,1] + 2*ξDT[mT,2]) 
            ξD_filtered[mT,nT] = (4*ξDT[mT-1,nT-1] + 2*ξDT[mT-1,nT] + 2*ξDT[mT,nT-1] + ξDT[mT,nT]) 

            for j ∈ 2:nq-1 
                for k ∈ 2:mq-1 
                    ξDhat_filtered[k,j] = Gconvolve(@view ξDhat[k-1:k+1,j-1:j+1])
                    # ξDhat_filtered[k,j] = sum(G .* ξDhat[k-1:k+1,j-1:j+1])
                end
            end
            for h ∈ 2:nq-1
                ξDhat_filtered[1,h] = Gconvolve12all(ξDhat[1:2,h-1:h+1])
                ξDhat_filtered[mq,h] = Gconvolve23all(ξDhat[mq-1:mq,h-1:h+1])
                # ξDhat_filtered[1,h] = sum(G12all .* ξDhat[1,h])
                # ξDhat_filtered[mq,h] = sum(G23all .* ξDhat[mq,h])
            end
            for v ∈ 2:mq-1
                ξDhat_filtered[v,1] = Gconvolveall12(ξDhat[v-1:v+1,1:2])
                ξDhat_filtered[v,nq] = Gconvolveall23(ξDhat[v-1:v+1,nq-1:nq])
                # ξDhat_filtered[v,1] = sum(Gall12 .* ξDhat[v,1])
                # ξDhat_filtered[v,nq] = sum(Gall23 .* ξDhat[v,nq])
            end 

            ξDhat_filtered[1,1] = (ξDhat[1,1] + 2*ξDhat[1,2] + 2*ξDhat[2,1] + 4*ξDhat[2,2])
            ξDhat_filtered[1,nq] = (2*ξDhat[1,nq-1] + ξDhat[1,nq] + 4*ξDhat[2,nq-1] + 2*ξDhat[2,nq])
            ξDhat_filtered[mq,1] = (2*ξDhat[mq-1,1] + 4*ξDhat[mq-1,2] + ξDhat[mq,1] + 2*ξDhat[mq,2])
            ξDhat_filtered[mq,nq] = (4*ξDhat[mq-1,nq-1] + 2*ξDhat[mq-1,nq] + 2*ξDhat[mq,nq-1] + ξDhat[mq,nq])
        end

        ∂x!(dξDdx, ξD_filtered)
        ∂y!(dξDhatdy, ξDhat_filtered)
        ∂x!(dtracedx, trace_filtered)

        ∂x!(dξDhatdx, ξDhat_filtered)
        ∂y!(dξDdy, ξD_filtered)
        ∂y!(dtracedy, trace_filtered)

    else 
        # compute final derivatives of everything with no filter 
        ∂x!(dξDdx, ξDT)
        ∂y!(dξDhatdy, ξDhat)
        ∂x!(dtracedx, ξsqT)

        ∂x!(dξDhatdx, ξDhat)
        ∂y!(dξDdy, ξDT)
        ∂y!(dtracedy, ξsqT)
    end

    s = Δ^2 * scale
    # for kj in eachindex(S_u,dξDdx,temp,dtracedx)
    for j ∈ 1:nuy
        for k ∈ 1:nux
            # S_u[k,j] = (-dξDdx[k,j] + dξDhatdy[k+1,j] + dtracedx[k,j]) / s
            S_u[k,j] = (-dξDdx[k,j] + dξDhatdy[k+1,j] + dtracedx[k,j]) / s
        end
    end

    # for kj in eachindex(S_v,dξDdy,dtracedy)
    for j ∈ 1:nvy
        for k ∈ 1:nvx
            # S_v[k,j] = (dξDhatdx[k,j+1] + dξDdy[k,j] + dtracedy[k,j]) / s
            S_v[k,j] = (dξDhatdx[k,j+1] + dξDdy[k,j] + dtracedy[k,j]) / s
        end
    end

end

# from a prior, different version of computing the S operators 

# @unpack D_n, D_nT, D_q = Diag.ZBVars

# ∂x!(dudx, u)
# ∂y!(dudy, u)

# ∂x!(dvdx, v)
# ∂y!(dvdy, v)

# mq,nq = size(ξ)
# mTh,nTh = size(Dhat)
# mT,nT = size(trace)

# @boundscheck (mq+2,nq+2) == size(dvdx) || throw(BoundsError())
# @boundscheck (mTh+1,nTh+1) == size(dvdx) || throw(BoundsError())
# @boundscheck (mq+2+ep,nq+2) == size(dudy) || throw(BoundsError())

# # Relative vorticity and shear deformation, cell corners
# @inbounds for j ∈ 1:nq
#     for k ∈ 1:mq
#         ξ[k,j] = dvdx[k+1,j+1] - dudy[k+1,j+1]
#     end
# end

# # Shear deformation, cell corners with halo (131,131)
# m_temp,n_temp = size(D_n)
# @inbounds for j ∈ 1:n_temp
#     for i ∈ 1:m_temp
#         D_n[i,j] = dudy[i+ep,j] + dvdx[i,j]
#     end
# end

# # Move to cell centers with halo (130,130)
# Ixy!(D_nT, D_n)

# # Stretch deformation, cell centers (with halo) (130,130)
# @inbounds for j ∈ 1:nTh
#     for k ∈ 1:mTh
#         Dhat[k,j] = dudx[k,j+1] - dvdy[k+1,j]
#     end
# end

# ξsq .= ξ.^2 

# # Last interpolation of D, moving to corners without halo
# Ixy!(D_q, D_nT)

# # Move ξ^2 to cell centers (128,128)
# Ixy!(ξsqT, ξsq)

# # Computing ξ ⋅ D and placing on cell centers 
# Ixy!(ξD, ξ .* D_q)

# # Computing ξ ⋅ Dhat, cell corners 
# Ixy!(Dhatq, Dhat)
# @inbounds for j ∈ 1:nq
#     for k ∈ 1:mq 
#         ξDhat[k,j] = ξ[k,j] * Dhatq[k,j]
#     end
# end

# # Computing final derivatives of everything
# ∂x!(dξDdx, ξD)
# ∂y!(dξDhatdy, ξDhat)
# ∂x!(dtracedx, ξsqT)

# ∂x!(dξDhatdx, ξDhat)
# ∂y!(dξDdy, ξD)
# ∂y!(dtracedy, ξsqT)

# temp = (dξDhatdy[2:end-1,:])
# s = Δ^2 * scale
# @inbounds for j ∈ 1:nuy
#     for k ∈ 1:nux
#         S_u[k,j] = κ_BC * (-dξDdx[k,j] + temp[k,j] + dtracedx[k,j]) / s
#     end
# end

# temp2 = dξDhatdx[:,2:end-1]
# @inbounds for j ∈ 1:nvy
#     for k ∈ 1:nvx
#         S_v[k,j] = κ_BC * (temp2[k,j] + dξDdy[k,j] + dtracedy[k,j]) / s
#     end
# end

# from prior attempt to implement a different version of κ_BC, will still have a tunable parameter
# made a mistake, this is largely for numerical stability and not for actually 
# correcting how much energy is injected into the model 
# @inbounds for j ∈ 1:nT 
#     for k ∈ 1:mT
#         γ[k,j] = γ₀ * (1 + (sqrt(trace[k,j])/abs(f₀)))^(-1)
#     end
# end
# Ix!(γ_u,γ)
# Iy!(γ_v,γ)

### for adding additional trace terms 
# @inbounds for j ∈ 1:nT 
#     for k ∈ 1:mT
#         trace[k,j] = ξpDT[k,j] + Dhatsq[k+1,j+1]
#     end
# end