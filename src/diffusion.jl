"""Transit function to call the specified diffusion scheme."""
function diffusion!(    u::AbstractMatrix,
                        v::AbstractMatrix,
                        Diag::DiagnosticVars,
                        S::ModelSetup)

    if S.parameters.diffusion == "constant"
        diffusion_constant!(u,v,Diag,S)
    elseif S.parameters.diffusion == "Smagorinsky"
        diffusion_smagorinsky!(u,v,Diag,S)
    else
        throw(error("Diffusion scheme $(S.parameters.diffusion) is unsupported."))
    end
end


"""Biharmonic diffusion operator with constant viscosity coefficient
Viscosity = ν∇⁴ ⃗u. Although constant, the coefficient is actually inside
Viscosity = ∇⋅ν∇∇² ⃗u."""
function diffusion_constant!(   u::AbstractMatrix,
                                v::AbstractMatrix,
                                Diag::DiagnosticVars,
                                S::ModelSetup)

    stress_tensor!(u,v,Diag)
    viscous_tensor_constant!(Diag,S)

    @unpack LLu1,LLu2,LLv1,LLv2 = Diag.Smagorinsky
    @unpack S11,S12,S21,S22 = Diag.Smagorinsky

    ∂x!(LLu1,S11)
    ∂y!(LLu2,S12)
    ∂x!(LLv1,S21)
    ∂y!(LLv2,S22)
end

""" Smagorinsky-like biharmonic viscosity
Viscosity = ∇ ⋅ (cSmag Δ⁴ |D| ∇∇² ⃗u)
The Δ⁴-scaling is omitted as gradient operators are dimensionless."""
function diffusion_smagorinsky!(u::AbstractMatrix,
                                v::AbstractMatrix,
                                Diag::DiagnosticVars,
                                S::ModelSetup)

    @unpack dudx,dvdy,dvdx,dudy = Diag.Vorticity

    ∂x!(dudx,u)
    ∂y!(dvdy,v)
    ∂x!(dvdx,v)
    ∂y!(dudy,u)

    # biharmonic diffusion
    stress_tensor!(u,v,Diag)
    smagorinsky_coeff!(Diag,S)
    viscous_tensor_smagorinsky!(Diag)

    @unpack LLu1,LLu2,LLv1,LLv2 = Diag.Smagorinsky
    @unpack S11,S12,S21,S22 = Diag.Smagorinsky

    ∂x!(LLu1,S11)
    ∂y!(LLu2,S12)
    ∂x!(LLv1,S21)
    ∂y!(LLv2,S22)
end

"""νSmag = cSmag * |D|, where deformation rate |D| = √((∂u/∂x - ∂v/∂y)^2 + (∂u/∂y + ∂v/∂x)^2).
The grid spacing Δ is omitted here as the operators are dimensionless."""
function smagorinsky_coeff!(Diag::DiagnosticVars,S::ModelSetup)

    @unpack dudx,dudy,dvdx,dvdy = Diag.Vorticity
    @unpack DS,DS_q,DT,νSmag,νSmag_q = Diag.Smagorinsky
    @unpack ep = S.grid
    @unpack cSmag = S.constants

    # horizontal shearing strain squared
    m,n = size(DS_q)
    @boundscheck (m+ep,n) == size(dudy) || throw(BoundsError())
    @boundscheck (m,n) == size(dvdx) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            DS_q[i,j] = (dudy[i+ep,j] + dvdx[i,j])^2
        end
    end

    Ixy!(DS,DS_q)

    # horizontal tension squared
    m,n = size(DT)
    @boundscheck (m+ep,n+2) == size(dudx) || throw(BoundsError())
    @boundscheck (m+2,n) == size(dvdy) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            DT[i,j] = (dudx[i+ep,j+1] + dvdy[i+1,j])^2
        end
    end

    # viscosity = Smagorinsky coefficient times deformation rate
    m,n = size(νSmag)
    @boundscheck (m,n) == size(DS) || throw(BoundsError())
    @boundscheck (m,n) == size(DT) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            νSmag[i,j] = cSmag*sqrt(DS[i,j] + DT[i,j])
        end
    end

    Ixy!(νSmag_q,νSmag)
end

"""Biharmonic stress tensor ∇∇²(u,v) = (∂/∂x(∇²u), ∂/∂y(∇²u); ∂/∂x(∇²v), ∂/∂y(∇²v))"""
function stress_tensor!(u::AbstractMatrix,v::AbstractMatrix,Diag::DiagnosticVars)
    @unpack Lu,Lv,dLudx,dLudy,dLvdx,dLvdy = Diag.Laplace
    ∇²!(Lu,u)
    ∇²!(Lv,v)
    ∂x!(dLudx,Lu)
    ∂y!(dLudy,Lu)
    ∂x!(dLvdx,Lv)
    ∂y!(dLvdy,Lv)
end

"""Biharmonic stress tensor times Smagorinsky coefficient
νSmag * ∇∇² ⃗u = (S11, S12; S21, S22)."""
function viscous_tensor_smagorinsky!(Diag::DiagnosticVars)

    @unpack dLudx,dLudy,dLvdx,dLvdy = Diag.Laplace
    @unpack νSmag,νSmag_q,S11,S12,S21,S22 = Diag.Smagorinsky
    @unpack ep = Diag.Smagorinsky

    m,n = size(S11)
    @boundscheck (m+2-ep,n) == size(νSmag) || throw(BoundsError())
    @boundscheck (m,n) == size(dLudx) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            S11[i,j] = νSmag[i+1-ep,j] * dLudx[i,j]
        end
    end

    m,n = size(S12)
    @boundscheck (m,n) == size(νSmag_q) || throw(BoundsError())
    @boundscheck (m+ep,n) == size(dLudy) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            S12[i,j] = νSmag_q[i,j] * dLudy[i+ep,j]
        end
    end

    m,n = size(S21)
    @boundscheck (m,n) == size(νSmag_q) || throw(BoundsError())
    @boundscheck (m,n) == size(dLvdx) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            S21[i,j] = νSmag_q[i,j] * dLvdx[i,j]
        end
    end

    m,n = size(S22)
    @boundscheck (m,n+2) == size(νSmag) || throw(BoundsError())
    @boundscheck (m,n) == size(dLvdy) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            S22[i,j] = νSmag[i,j+1] * dLvdy[i,j]
        end
    end
end

"""Biharmonic stress tensor times constant viscosity coefficient
νB * ∇∇² ⃗u = (S11, S12; S21, S22)"""
function viscous_tensor_constant!(  Diag::DiagnosticVars,
                                    S::ModelSetup)

    @unpack dLudx,dLudy,dLvdx,dLvdy = Diag.Laplace
    @unpack S11,S12,S21,S22 = Diag.Smagorinsky
    @unpack ep = S.grid
    @unpack νB = S.constants

    m,n = size(S11)
    @boundscheck (m,n) == size(dLudx) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            S11[i,j] = νB * dLudx[i,j]
        end
    end

    m,n = size(S12)
    @boundscheck (m+ep,n) == size(dLudy) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            S12[i,j] = νB * dLudy[i+ep,j]
        end
    end

    m,n = size(S21)
    @boundscheck (m,n) == size(dLvdx) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            S21[i,j] = νB * dLvdx[i,j]
        end
    end

    m,n = size(S22)
    @boundscheck (m,n) == size(dLvdy) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            S22[i,j] = νB * dLvdy[i,j]
        end
    end
end

"""Update u with bottom friction tendency (Bu,Bv) and biharmonic viscosity."""
function add_drag_diff_tendencies!( u::Matrix{Tprog},
                                    v::Matrix{Tprog},
                                    Diag::DiagnosticVars{T,Tprog},
                                    S::ModelSetup{T,Tprog}) where {T,Tprog}

    @unpack Bu,Bv = Diag.Bottomdrag
    @unpack LLu1,LLu2,LLv1,LLv2 = Diag.Smagorinsky
    @unpack halo,ep,Δt_diff = S.grid
    @unpack compensated = S.parameters
    @unpack du_comp,dv_comp = Diag.Tendencies

    m,n = size(u) .- (2*halo,2*halo)
    @boundscheck (m+2-ep,n+2) == size(Bu) || throw(BoundsError())
    @boundscheck (m,n+2) == size(LLu1) || throw(BoundsError())
    @boundscheck (m+2-ep,n) == size(LLu2) || throw(BoundsError())

    if S.parameters.zb_forcing_dissipation
        ZB_momentum(u,v,S,Diag)
    end

    if S.parameters.nn_forcing_dissipation
        if S.parameters.handwritten
            handwritten_NN_momentum(u,v,S)
        else
            NN_momentum(u,v,S)
        end
    end 

    if compensated
        @inbounds for j ∈ 1:n
            for i ∈ 1:m
                du = Δt_diff*convert(Tprog,Bu[i+1-ep,j+1] + LLu1[i,j+1] + LLu2[i+1-ep,j]) - du_comp[i+2,j+2]
                u_new = u[i+2,j+2] + du
                du_comp[i+2,j+2] = (u_new - u[i+2,j+2]) - du
                u[i+2,j+2] = u_new
            end
        end
    elseif S.parameters.zb_forcing_dissipation
        @inbounds for j ∈ 1:n
            for i ∈ 1:m
                u[i+2,j+2] += Δt_diff*(Tprog(Bu[i+1-ep,j+1]) + Tprog(LLu1[i,j+1]) + Tprog(LLu2[i+1-ep,j]) + Tprog(Diag.ZBVars.S_u[i,j]))
            end
        end
    elseif S.parameters.nn_forcing_dissipation
        @inbounds for j ∈ 1:n
            for i ∈ 1:m
                u[i+2,j+2] += Δt_diff*(Tprog(Bu[i+1-ep,j+1]) + Tprog(LLu1[i,j+1]) + Tprog(LLu2[i+1-ep,j]) + Tprog(Diag.NNVars.S_u[i,j]))
            end
        end
    else
        @inbounds for j ∈ 1:n
            for i ∈ 1:m
                u[i+2,j+2] += Δt_diff*(Tprog(Bu[i+1-ep,j+1]) + Tprog(LLu1[i,j+1]) + Tprog(LLu2[i+1-ep,j]))
            end
        end
    end

    m,n = size(v) .- (2*halo,2*halo)
    @boundscheck (m+2,n+2) == size(Bv) || throw(BoundsError())
    @boundscheck (m,n+2) == size(LLv1) || throw(BoundsError())
    @boundscheck (m+2,n) == size(LLv2) || throw(BoundsError())

    if compensated
        @inbounds for j ∈ 1:n
            for i ∈ 1:m 
                dv = Δt_diff*convert(Tprog,Bv[i+1,j+1] + LLv1[i,j+1] + LLv2[i+1,j]) - dv_comp[i+2,j+2]
                v_new = v[i+2,j+2] + dv
                dv_comp[i+2,j+2] = (v_new - v[i+2,j+2]) - dv
                v[i+2,j+2] = v_new
            end
        end
    elseif S.parameters.zb_forcing_dissipation
        @inbounds for j ∈ 1:n
            for i ∈ 1:m
                v[i+2,j+2] += Δt_diff*(Tprog(Bv[i+1,j+1]) + Tprog(LLv1[i,j+1]) + Tprog(LLv2[i+1,j]) + Tprog(Diag.ZBVars.S_v[i,j]))
            end
        end
    elseif S.parameters.nn_forcing_dissipation
        @inbounds for j ∈ 1:n
            for i ∈ 1:m
                v[i+2,j+2] += Δt_diff*(Tprog(Bv[i+1,j+1]) + Tprog(LLv1[i,j+1]) + Tprog(LLv2[i+1,j]) + Tprog(Diag.NNVars.S_v[i,j]))
            end
        end
    else
        @inbounds for j ∈ 1:n
            for i ∈ 1:m
                v[i+2,j+2] += Δt_diff*(Tprog(Bv[i+1,j+1]) + Tprog(LLv1[i,j+1]) + Tprog(LLv2[i+1,j]))
            end
        end
    end

end
