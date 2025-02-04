"""Runge Kutta time stepping scheme diagnostic cariables collected in a struct."""
@with_kw struct RungeKuttaVars{T<:AbstractFloat,  ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end  # is there a u-point on the left edge?

    u0::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo)     # u-velocities for RK updates
    u1::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo)
    v0::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo)     # v-velocities for RK updates
    v1::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo)
    η0::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)     # sea surface height for RK updates
    η1::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)
end

"""Generator function for RungeKutta VarCollection."""
function RungeKuttaVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return RungeKuttaVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###################################################

"""Tendencies collected in a struct."""
@with_kw struct TendencyVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end  # is there a u-point on the left edge?

    du::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo)     # tendency of u without time step
    dv::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo)     # tendency of v without time step
    dη::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)     # tendency of η without time step

    # sum of tendencies (incl time step) over all sub-steps
    du_sum::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo) 
    dv_sum::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo)
    dη_sum::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)

    # compensation for tendencies (variant of Kahan summation)
    du_comp::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo) 
    dv_comp::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo)
    dη_comp::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)
end

"""Generator function for Tendencies VarCollection."""
function TendencyVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return TendencyVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###########################################################

"""VolumeFluxes collected in a struct."""
@with_kw struct VolumeFluxVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    h::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)         # layer thickness
    h_u::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη)     # layer thickness on u-grid
    U::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη)       # U=uh volume flux

    h_v::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη-1)     # layer thickness on v-grid
    V::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη-1)       # V=vh volume flux

    dUdx::ArrayTy = zeros(T,nx+2*haloη-2,ny+2*haloη)    # gradients thereof
    dVdy::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη-2)
end

"""Generator function for VolumeFluxes VarCollection."""
function VolumeFluxVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return VolumeFluxVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###############################################################

"""Vorticity variables collected in a struct."""
@with_kw struct VorticityVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    h_q::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-1)  # layer thickness h interpolated on q-grid
    q::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-1)    # potential vorticity

    q_v::ArrayTy = zeros(T,nx+2*haloη-2,ny+2*haloη-1)  # q interpolated on v-grid
    U_v::ArrayTy = zeros(T,nx+2*haloη-2,ny+2*haloη-1)  # mass flux U=uh on v-grid

    q_u::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-2)  # q interpolated on u-grid
    V_u::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-2)  # mass flux V=vh on v-grid

    qhu::ArrayTy = zeros(T,nvx,nvy)            # potential vorticity advection term u-component
    qhv::ArrayTy = zeros(T,nux,nuy)            # potential vorticity advection term v-component

    u_v::ArrayTy = zeros(T,nux+2*halo-1,nuy+2*halo-1)  # u-velocity on v-grid
    v_u::ArrayTy = zeros(T,nvx+2*halo-1,nvy+2*halo-1)  # v-velocity on u-grid

    dudx::ArrayTy = zeros(T,nux+2*halo-1,nuy+2*halo)   # ∂u/∂x
    dudy::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo-1)   # ∂u/∂y

    dvdx::ArrayTy = zeros(T,nvx+2*halo-1,nvy+2*halo)   # ∂v/∂x
    dvdy::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo-1)   # ∂v/∂y
end

"""Generator function for Vorticity VarCollection."""
function VorticityVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return VorticityVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Bernoulli variables collected in a struct."""
@with_kw struct BernoulliVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    u²::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo)         # u-velocity squared
    v²::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo)         # v-velocity squared

    KEu::ArrayTy = zeros(T,nux+2*halo-1,nuy+2*halo)      # u-velocity squared on T-grid
    KEv::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo-1)      # v-velocity squared on T-grid

    p::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)          # Bernoulli potential
    dpdx::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη)     # ∂p/∂x
    dpdy::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη-1)     # ∂p/∂y
end

"""Generator function for Bernoulli VarCollection."""
function BernoulliVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return BernoulliVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Bottomdrag variables collected in a struct."""
@with_kw struct BottomdragVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    sqrtKE::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)       # sqrt of kinetic energy
    sqrtKE_u::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη)   # interpolated on u-grid
    sqrtKE_v::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη-1)   # interpolated on v-grid

    Bu::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη)         # bottom friction term u-component
    Bv::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη-1)         # bottom friction term v-component
end

"""Generator function for Bottomdrag VarCollection."""
function BottomdragVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return BottomdragVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""ArakawaHsu variables collected in a struct."""
@with_kw struct ArakawaHsuVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    # Linear combination of potential vorticity
    qα::ArrayTy = zeros(T,nx+2*haloη-2,ny+2*haloη-2)
    qβ::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-2)
    qγ::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-2)
    qδ::ArrayTy = zeros(T,nx+2*haloη-2,ny+2*haloη-2)
end

"""Generator function for ArakawaHsu VarCollection."""
function ArakawaHsuVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return ArakawaHsuVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Laplace variables collected in a struct."""
@with_kw struct LaplaceVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    Lu::ArrayTy = zeros(T,nux+2*halo-2,nuy+2*halo-2)         # ∇²u
    Lv::ArrayTy = zeros(T,nvx+2*halo-2,nvy+2*halo-2)         # ∇²v

    # Derivatives of Lu,Lv
    dLudx::ArrayTy = zeros(T,nux+2*halo-3,nuy+2*halo-2)
    dLudy::ArrayTy = zeros(T,nux+2*halo-2,nuy+2*halo-3)
    dLvdx::ArrayTy = zeros(T,nvx+2*halo-3,nvy+2*halo-2)
    dLvdy::ArrayTy = zeros(T,nvx+2*halo-2,nvy+2*halo-3)
end

"""Generator function for Laplace VarCollection."""
function LaplaceVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return LaplaceVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Smagorinsky variables collected in a struct."""
@with_kw struct SmagorinskyVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    DT::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)       # Tension squared (on the T-grid)
    DS::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)       # Shearing strain squared (on the T-grid)
    νSmag::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)    # Viscosity coefficient

    # Tension squared on the q-grid
    DS_q::ArrayTy = zeros(T,nvx+2*halo-1,nvy+2*halo)

    # Smagorinsky viscosity coefficient on the q-grid
    νSmag_q::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-1)

    # Entries of the Smagorinsky viscous tensor
    S12::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-1)
    S21::ArrayTy = zeros(T,nx+2*haloη-1,ny+2*haloη-1)

    S11::ArrayTy = zeros(T,nux+2*halo-3,nuy+2*halo-2)
    S22::ArrayTy = zeros(T,nvx+2*halo-2,nvy+2*halo-3)

    # u- and v-components 1 and 2 of the biharmonic diffusion tendencies
    LLu1::ArrayTy = zeros(T,nux+2*halo-4,nuy+2*halo-2)
    LLu2::ArrayTy = zeros(T,nx+1,ny)

    LLv1::ArrayTy = zeros(T,nx,ny+1)
    LLv2::ArrayTy = zeros(T,nvx+2*halo-2,nvy+2*halo-4)
end

"""Generator function for Smagorinsky VarCollection."""
function SmagorinskyVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return SmagorinskyVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""SemiLagrange variables collected in a struct."""
@with_kw struct SemiLagrangeVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int
    halosstx::Int
    halossty::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end   # u-grid in x-direction
    nuy::Int = ny                                       # u-grid in y-direction
    nvx::Int = nx                                       # v-grid in x-direction
    nvy::Int = ny-1                                     # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end   # q-grid in x-direction
    nqy::Int = ny+1                                     # q-grid in y-direction

    # EDGE POINT (1 = yes, 0 = no)
    ep::Int = if bc == "periodic" 1 else 0 end      # is there a u-point on the left edge?

    xd::ArrayTy = zeros(T,nx,ny)                         # departure points x-coord
    yd::ArrayTy = zeros(T,nx,ny)                         # departure points y-coord

    um::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo)         # u-velocity temporal mid-point
    vm::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo)         # v-velocity temporal mid-point

    u_T::ArrayTy = zeros(T,nux+2*halo-1,nuy+2*halo)      # u-velocity interpolated on T-grid
    um_T::ArrayTy = zeros(T,nux+2*halo-1,nuy+2*halo)     # um interpolated on T-grid
    v_T::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo-1)      # v-velocity interpolated on T-grid
    vm_T::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo-1)     # vm interpolated on T-grid

    uinterp::ArrayTy = zeros(T,nx,ny)                    # u interpolated on mid-point xd,yd
    vinterp::ArrayTy = zeros(T,nx,ny)                    # v interpolated on mid-point xd,yd

    ssti::ArrayTy = zeros(T,nx+2*halosstx,ny+2*halossty) # sst interpolated on departure points
    sst_ref::ArrayTy = zeros(T,nx+2*halosstx,ny+2*halossty) # sst initial conditions for relaxation

    # compensated summation
    dsst_comp::ArrayTy = zeros(T,nx+2*halosstx,ny+2*halossty)
end

"""Generator function for SemiLagrange VarCollection."""
function SemiLagrangeVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G
    @unpack halosstx,halossty = G

    return SemiLagrangeVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη,
                            halosstx=halosstx,halossty=halossty)
end

###################################################################

""" Variables that appear in Zanna-Bolton forcing term """ 
@with_kw struct ZBVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int
    halosstx::Int
    halossty::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end      # u-grid in x-direction
    nuy::Int = ny                                          # u-grid in y-direction
    nvx::Int = nx                                          # v-grid in x-direction
    nvy::Int = ny-1                                        # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end      # q-grid in x-direction
    nqy::Int = ny+1                                        # q-grid in y-direction

    dudx::ArrayTy = zeros(T,nux+2*halo-1,nuy+2*halo)    # ∂u/∂x
    dudy::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo-1)    # ∂u/∂y
    dvdx::ArrayTy = zeros(T,nvx+2*halo-1,nvy+2*halo)    # ∂v/∂x
    dvdy::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo-1)    # ∂v/∂y

    # these are only utilized in a scheme where γ varies spacially
    γ::ArrayTy = zeros(T,nx,ny)
    γ_u::ArrayTy = zeros(T,nux,nuy)
    γ_v::ArrayTy = zeros(T,nvx,nvy)

    Ker::ArrayTy = zeros(3,3)    # convolutional kernal

    ζ::ArrayTy = zeros(T,nqx,nqy)      # relative vorticity, cell corners 
    ζsq::ArrayTy = zeros(T,nqx,nqy)    # relative vorticity squared, cell corners 

    D::ArrayTy = zeros(T,nqx,nqy)      # shear deformation of flow field, cell corners 
    Dsq::ArrayTy = zeros(T,nqx,nqy)    # square of the tensor 

    D_n::ArrayTy = zeros(T,nvx+2*halo-1,nvy+2*halo)
    D_nT::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη) 
    D_q::ArrayTy = zeros(T,nqx,nqy)

    Dhat::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)     # stretch deformation of flow field, cell centers w/ halo
    Dhatsq::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)   # square of the tensor
    Dhatq::ArrayTy = zeros(T,nqx,nqy)                  # tensor interpolated onto q-grid

    ζpDT::ArrayTy = zeros(T,nx,ny)     # ζ^2 + D^2 interpolated to cell centers, not currently used
    ζsqT::ArrayTy = zeros(T,nx,ny)     # ζ^2 interpolated to cell centers
    ζD::ArrayTy = zeros(T,nqx,nqy)     # ζ ⋅ D, cell corners
    ζDT::ArrayTy = zeros(T,nx,ny)      # ζ ⋅ D, placed on cell centers
    ζDhat::ArrayTy = zeros(T,nqx,nqy)  # ζ ⋅ Dhat, cell corners
    
    trace::ArrayTy = zeros(T,nx,ny)     # ξ^2 + D^2 + Dhat^2, cell centers

    ζD_filtered::ArrayTy = zeros(T,nx,ny)      # ξD with filter applied
    ζDhat_filtered::ArrayTy = zeros(T,nqx,nqy)   # ξDhat with filter applied
    trace_filtered::ArrayTy = zeros(T,nx,ny)     # trace with filter applied

    dζDdx::ArrayTy = zeros(T,nux,nuy)             # u-grid
    dζDhatdy::ArrayTy = zeros(T,nux+halo,nuy)     # u-grid, initially with extra halo points
    dtracedx::ArrayTy = zeros(T,nux,nuy)          # u-grid 

    S_u::ArrayTy = zeros(T,nux,nuy)             # total forcing in x-direction

    dζDhatdx::ArrayTy = zeros(T,nvx,nvy+halo)   # v-grid, initially with extra halo points
    dζDdy::ArrayTy = zeros(T,nvx,nvy)           # v-grid
    dtracedy::ArrayTy = zeros(T,nvx,nvy)        # v-grid

    S_v::ArrayTy = zeros(T,nvx,nvy)             # total forcing in y-direction

end

"""Generator function for ZB_momentum terms."""
function ZBVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G
    @unpack halosstx,halossty = G

    Ker = zeros(3,3)
    Ker[1,1] = 1 
    Ker[1,2] = 2 
    Ker[1,3] = 1
    Ker[2,1] = 2 
    Ker[2,2] = 4 
    Ker[2,3] = 2 
    Ker[3,1] = 1 
    Ker[3,2] = 2 
    Ker[3,3] = 1 

    return ZBVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη,
                            halosstx=halosstx,halossty=halossty,Ker=Ker)
end

###########################################################################################

""" Variables that appear in NN forcing term """
@with_kw mutable struct NNVars{T<:AbstractFloat, ArrayTy}

    # to be specified
    nx::Int
    ny::Int
    bc::String
    halo::Int
    haloη::Int
    halosstx::Int
    halossty::Int

    nux::Int = if (bc == "periodic") nx else nx-1 end      # u-grid in x-direction
    nuy::Int = ny                                          # u-grid in y-direction
    nvx::Int = nx                                          # v-grid in x-direction
    nvy::Int = ny-1                                        # v-grid in y-direction
    nqx::Int = if (bc == "periodic") nx else nx+1 end      # q-grid in x-direction
    nqy::Int = ny+1                                        # q-grid in y-direction

    dudx::ArrayTy = zeros(T,nux+2*halo-1,nuy+2*halo)    # ∂u/∂x
    dudy::ArrayTy = zeros(T,nux+2*halo,nuy+2*halo-1)    # ∂u/∂y
    dvdx::ArrayTy = zeros(T,nvx+2*halo-1,nvy+2*halo)    # ∂v/∂x
    dvdy::ArrayTy = zeros(T,nvx+2*halo,nvy+2*halo-1)    # ∂v/∂y

    # weights_corner is applied to ζ and D to form the diagonal terms of S,
    # and weights_center is applied to ζ and D̃ to form the off-diagonal terms
    # of S. Initially this will just be a single layer to see if we can get the model
    # with the neural net running
    # the initial weight values might need to change, for now I'm setting them to zero
    weights_corner::ArrayTy = zeros(T,2,22)
    corner_outdim::Int = 2
    corner_indim::Int = 22

    weights_center::ArrayTy = zeros(T,1,17)
    center_outdim::Int = 1
    center_indim::Int = 17

    ζ::ArrayTy = zeros(T,nqx,nqy)      # relative vorticity, cell corners 

    D::ArrayTy = zeros(T,nqx,nqy)      # shear deformation of flow field, cell corners 

    Dhat::ArrayTy = zeros(T,nx+2*haloη,ny+2*haloη)     # stretch deformation of flow field, cell centers w/ halo

    ζT::ArrayTy = zeros(T,nx,ny)         # ζ interpolated to cell centers
    DT::ArrayTy = zeros(T,nx,ny)         # D, interpolated on cell centers
    ζDhat::ArrayTy = zeros(T,nqx,nqy)    # ζ ⋅ Dhat, cell corners

    T11::ArrayTy = zeros(T,nqx,nqy)
    T12::ArrayTy = zeros(T,nx,ny)
    T21::ArrayTy = zeros(T,nx,ny)
    T22::ArrayTy = zeros(T,nqx,nqy)

    S_u::ArrayTy = zeros(T,nux,nuy)             # total forcing in x-direction
    S_v::ArrayTy = zeros(T,nvx,nvy)             # total forcing in y-direction

end

"""Generator function for NN momentum terms."""
function NNVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G
    @unpack halosstx,halossty = G

    return NNVars{T, Array{T, 2}}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη,
                            halosstx=halosstx,halossty=halossty
    )
end

##########################################################################################

"""Preallocate the diagnostic variables and return them as matrices in structs."""
function preallocate(   ::Type{T},
                        ::Type{Tprog},
                        G::Grid
                    ) where {T<:AbstractFloat,Tprog<:AbstractFloat}

    RK = RungeKuttaVars{Tprog}(G)
    TD = TendencyVars{Tprog}(G)
    VF = VolumeFluxVars{T}(G)
    VT = VorticityVars{T}(G)
    BN = BernoulliVars{T}(G)
    BD = BottomdragVars{T}(G)
    AH = ArakawaHsuVars{T}(G)
    LP = LaplaceVars{T}(G)
    SM = SmagorinskyVars{T}(G)
    SL = SemiLagrangeVars{T}(G)
    PV = PrognosticVars{T}(G)
    ZB = ZBVars{T}(G)
    NN = NNVars{T}(G)

    return DiagnosticVars{T,Tprog}(RK,TD,VF,VT,BN,BD,AH,LP,SM,SL,PV,ZB,NN)
end
