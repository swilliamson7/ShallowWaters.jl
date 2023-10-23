"""
This function will compute an additional forcing term that will (hopefully) act
as the eddy parameterization. I'm doing this by 
(1) adding a new structure that will preallocate all of the operators I need (called ZB_momentum)
(2) I compute the main operators that appear in the new term: ξ, D, Dhat. ξ is the relative vorticity,
D is the shear deformation of the flow field (both of these live on cell corners), and Dhat is the stretch
deformation of the flow field (this lives on the cell centers)
(3) Using ξ, D, and Dhat I then compute the individual pieces of the forcing that appear, comments
below explain on which grid they live and so on 
"""
function ZB_momentum(u, v, S, Diag) 

    @unpack γ₀ = S.parameters
    @unpack ξ, ξsq, D, Dsq, Dhat, Dhatsq, Dhatq = Diag.ZBVars
    @unpack ξD, ξDT, ξDhat, ξsqT, trace = Diag.ZBVars
    @unpack ξpDT = Diag.ZBVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.ZBVars

    @unpack dξDdx, dξDhatdy, dtracedx = Diag.ZBVars
    @unpack dξDhatdx, dξDdy, dtracedy = Diag.ZBVars
    @unpack S_u, S_v = Diag.ZBVars
    @unpack Δ, scale, f₀ = S.grid

    @unpack γ, γ_u, γ_v = Diag.ZBVars
    @unpack G = Diag.ZBVars

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
    
    # ξsq .= (κ_BC / 2) .* ξ.^2 
    ξsq .= ξ.^2 
    Dsq .= D.^2 
    Dhatsq .= Dhat.^2 

    # Trace computation (second term in forcing term)
    # Computing the sum of ξ^2 and D^2 and moving to cell centers
    Ixy!(ξpDT, ξsq + Dsq)
    Ixy!(ξsqT, ξsq)
    @inbounds for j ∈ 1:nT 
        for k ∈ 1:mT
            trace[k,j] = ξpDT[k,j] + Dhatsq[k+1,j+1]
        end
    end

    # Computing ξ ⋅ D and placing on cell centers 
    ξD .= ξ .* D
    Ixy!(ξDT, ξD)
    # ξDT .= κ_BC .* ξDT

    # Computing ξ ⋅ Dhat, cell corners 
    Ixy!(Dhatq, Dhat)
    @inbounds for j ∈ 1:nq
        for k ∈ 1:mq 
            ξDhat[k,j] = ξ[k,j] * Dhatq[k,j]
        end
    end
    # for kj in eachindex(ξDhat,ξ,Dhatq) 
    #     # ξDhat[kj] = κ_BC * ξ[kj] * Dhatq[kj]
    #     ξDhat[kj] = ξ[kj] * Dhatq[kj]
    # end

    # Computing final derivatives of everything
    ∂x!(dξDdx, ξDT)
    ∂y!(dξDhatdy, ξDhat)
    ∂x!(dtracedx, ξsqT)

    ∂x!(dξDhatdx, ξDhat)
    ∂y!(dξDdy, ξDT)
    ∂y!(dtracedy, ξsqT)

    # temp = @view dξDhatdy[2:end-1,:]
    s = Δ^2 * scale
    # for kj in eachindex(S_u,dξDdx,temp,dtracedx)
    for j ∈ 1:nuy
        for k ∈ 1:nux
            # S_u[k,j] = (-dξDdx[k,j] + dξDhatdy[k+1,j] + dtracedx[k,j]) / s
            S_u[k,j] = κ_BC * (-dξDdx[k,j] + dξDhatdy[k+1,j] + dtracedx[k,j]) / s
        end
    end

    # for kj in eachindex(S_v,dξDdy,dtracedy)
    for j ∈ 1:nvy
        for k ∈ 1:nvx
            # S_v[k,j] = (dξDhatdx[k,j+1] + dξDdy[k,j] + dtracedy[k,j]) / s
            S_v[k,j] = κ_BC * (dξDhatdx[k,j+1] + dξDdy[k,j] + dtracedy[k,j]) / s
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