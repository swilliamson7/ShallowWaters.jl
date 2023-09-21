"""
This function will compute an additional forcing term that will (hopefully) acceleration
as the eddy parameterization. I'm doing this by 
(1) adding a new structure that will preallocate all of the operators I need (called ZB_momentum)
(2) I compute the main operators that appear in the new term: ξ, D, Dhat. ξ is the relative vorticity,
D is the shear deformation of the flow field (both of these live on cell corners), and Dhat is the stretch
deformation of the flow field (this lives on the cell centers)
(3) Using ξ, D, and Dhat I then compute the individual pieces of the forcing that appear, comments
below explain on which grid they live and so on 
"""
function ZB_momentum(u, v, S, Diag) 

    @unpack κ_BC = Diag.ZBVars
    @unpack ξ, ξsq, D, Dsq, Dhat, Dhatsq, Dhatq = Diag.ZBVars
    @unpack ξD, ξDhat, ξpDT, trace = Diag.ZBVars
    @unpack dudx, dudy, dvdx, dvdy = Diag.ZBVars

    @unpack dξDdx, dξDhatdy, dtracedx = Diag.ZBVars
    @unpack dξDhatdx, dξDdy, dtracedy = Diag.ZBVars
    @unpack S_u, S_v = Diag.ZBVars

    @unpack halo, haloη, ep, nux, nuy, nvx, nvy = S.grid

    ∂x!(dudx, u)
    ∂y!(dudy, u)

    ∂x!(dvdx, v)
    ∂y!(dvdy, v)

    mq,nq = size(ξ)
    mTh,nTh = size(Dhat)
    mT,nT = size(trace)

    @boundscheck (mq+2,nq+2) == size(dvdx) || throw(BoundsError())
    @boundscheck (mTh+1,nTh+1) == size(dvdx) || throw(BoundsError())
    @boundscheck (mq+2+ep,nq+2) == size(dudy) || throw(BoundsError())
    
    # Relative vorticity and shear deformation, cell corners
    @inbounds for j ∈ 1:nq
        for k ∈ 1:mq
            ξ[k,j] = dvdx[k+1,j+1] - dudy[k+1+ep,j+1]
            D[k,j] = dudy[k+1+ep,j+1] + dvdx[k+1,j+1]
        end
    end

    # Stretch deformation, cell centers (with halo)
    @inbounds for j ∈ 1:nTh
        for k ∈ 1:mTh
            Dhat[k,j] = dudx[k,j+1] - dvdy[k+1,j]
        end
    end
    
    ξsq .= ξ.^2 
    Dsq .= D.^2 
    Dhatsq .= Dhatsq.^2 

    # Trace computation(second term in forcing term)
    # Computing the sum of ξ^2 and D^2 and moving to cell centers
    Ixy!(ξpDT, ξsq + Dsq)
    one_half = convert(T,0.5)
    @inbounds for j ∈ 1:nT 
        for k ∈ 1:mT
            trace[k,j] = one_half * (ξpDT[k,j] + Dhatsq[k+1,j+1])
        end
    end

    # Computing ξ ⋅ D and placing on cell centers 
    Ixy!(ξD, ξ .* D)

    # Computing ξ ⋅ Dhat, cell corners 
    Ixy!(Dhatq, Dhat)
    @inbounds for j ∈ 1:nq
        for k ∈ 1:mq 
            ξDhat[k,j] = ξ[k,j] * Dhatq[k,j]
        end
    end

    # Computing final derivatives of everything
    ∂x!(dξDdx, ξD)
    ∂y!(dξDhatdy, ξDhat)
    ∂x!(dtracedx, trace)

    ∂x!(dξDhatdx, ξDhat)
    ∂y!(dξDdy, ξD)
    ∂y!(dtracedy, trace)

    temp = dξDhatdy[2:end-1,:]
    @inbounds for j ∈ 1:nuy
        for k ∈ 1:nux
            S_u[k,j] = κ_BC * (dξDdx[k,j] + temp[k,j] + dtracedx[k,j])
        end
    end

    temp2 = dξDhatdx[:,2:end-1]
    @inbounds for j ∈ 1:nvy
        for k ∈ 1:nvx
            S_v[k,j] = κ_BC * (temp2[k,j] + dξDdy[k,j] + dtracedy[k,j])
        end
    end
    
    # # relative vorticity, cell corners 
    # ξ .= dvdx - dudy
    # ξsq .= ξ.^2 

    # # shear deformation of flow, cell corners 
    # D .= dudy + dvdx
    # Dsq .= D.^2

    # # stretch deformation of flow, cell centers 
    # Dhat .= dudx[:, 2:end-1] - dvdy[2:end-1, :]
    # Dhatsq .= Dhat.^2

    # # Now computing the trace (second term in Eq. (6) of Zanna and Bolton (2020))
    # # Since ξsq and Dsq live on the cell corners, I first move their sum to cell centers 
    # # then add Dhatsq. The trace subsequently lives in cell centers 
    # Ixy!(ξpDT, ξsq + Dsq)
    # trace = 0.5 * (ξpDT + Dhatsq)

    # # The diagonal of S is ξD, which lives on cell corners, here I compute it and then 
    # # move it to cell centers 
    # Ixy!(ξD, ξ .* D)

    # # Dhat is a weird term, as the components of it (u_x and v_y) are different sizes 
    # # First I move Dhat to cell corners, then pad it with zeros so that I may compute 
    # # the off-diagonal of S, ξDhat (now living on cell corners)
    # Ixy!(Dhatq, Dhat)
    # m,n = size(Dhatq)
    # ξDhat .= ξ .* [zeros(T, 1, n+2); zeros(m, 1) Dhatq zeros(m, 1); zeros(1, n+2)]

    # # When the final derivatives I need to remove the halo to avoid 
    # # dimension mismatches
    # # u-grid 
    # ∂x!(dξDdx[halo+1:end-halo, halo+1:end-halo], ξD[haloη+1:end-haloη,haloη+1:end-haloη])
    # # u-grid  
    # ∂y!(dξDhatdy,ξDhat)
    # # u-grid  
    # ∂x!(dtracedx,trace)

    # S_u = - dξDdx + dξDhatdy + dtracedx

    # m,n = size(Dhatq)
    # ∂x!(dξDhatdx, [zeros(1, n+2); ξDhat; zeros(1, n+2)])
    # ∂y!(dξDdy, ξD)
    # ∂y!(dtracedy, trace)

    # S_v = dξDhatdx + dξDdy + dtracedy

end

