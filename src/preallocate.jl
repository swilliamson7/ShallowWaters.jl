"""Runge Kutta time stepping scheme diagnostic cariables collected in a struct."""
@with_kw struct RungeKuttaVars{T<:AbstractFloat}

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

    u0::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)     # u-velocities for RK updates
    u1::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)
    v0::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)     # v-velocities for RK updates
    v1::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)
    η0::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)     # sea surface height for RK updates
    η1::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)
end

"""Generator function for RungeKutta VarCollection."""
function RungeKuttaVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return RungeKuttaVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###################################################

"""Tendencies collected in a struct."""
@with_kw struct TendencyVars{T<:AbstractFloat}

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

    du::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)     # tendency of u without time step
    dv::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)     # tendency of v without time step
    dη::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)     # tendency of η without time step

    # sum of tendencies (incl time step) over all sub-steps
    du_sum::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo) 
    dv_sum::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)
    dη_sum::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)

    # compensation for tendencies (variant of Kahan summation)
    du_comp::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo) 
    dv_comp::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)
    dη_comp::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)
end

"""Generator function for Tendencies VarCollection."""
function TendencyVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return TendencyVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###########################################################

"""VolumeFluxes collected in a struct."""
@with_kw struct VolumeFluxVars{T<:AbstractFloat}

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

    h::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)         # layer thickness
    h_u::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)     # layer thickness on u-grid
    U::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)       # U=uh volume flux

    h_v::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)     # layer thickness on v-grid
    V::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)       # V=vh volume flux

    dUdx::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη)    # gradients thereof
    dVdy::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-2)
end

"""Generator function for VolumeFluxes VarCollection."""
function VolumeFluxVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return VolumeFluxVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

###############################################################

"""Vorticity variables collected in a struct."""
@with_kw struct VorticityVars{T<:AbstractFloat}

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

    h_q::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)  # layer thickness h interpolated on q-grid
    q::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)    # potential vorticity

    q_v::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη-1)  # q interpolated on v-grid
    U_v::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη-1)  # mass flux U=uh on v-grid

    q_u::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-2)  # q interpolated on u-grid
    V_u::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-2)  # mass flux V=vh on v-grid

    qhu::Array{T,2} = zeros(T,nvx,nvy)            # potential vorticity advection term u-component
    qhv::Array{T,2} = zeros(T,nux,nuy)            # potential vorticity advection term v-component

    u_v::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo-1)  # u-velocity on v-grid
    v_u::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo-1)  # v-velocity on u-grid

    dudx::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)   # ∂u/∂x
    dudy::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo-1)   # ∂u/∂y

    dvdx::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo)   # ∂v/∂x
    dvdy::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)   # ∂v/∂y
end

"""Generator function for Vorticity VarCollection."""
function VorticityVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return VorticityVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Bernoulli variables collected in a struct."""
@with_kw struct BernoulliVars{T<:AbstractFloat}

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

    u²::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)         # u-velocity squared
    v²::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)         # v-velocity squared

    KEu::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)      # u-velocity squared on T-grid
    KEv::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)      # v-velocity squared on T-grid

    p::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)          # Bernoulli potential
    dpdx::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)     # ∂p/∂x
    dpdy::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)     # ∂p/∂y
end

"""Generator function for Bernoulli VarCollection."""
function BernoulliVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return BernoulliVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Bottomdrag variables collected in a struct."""
@with_kw struct BottomdragVars{T<:AbstractFloat}

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

    sqrtKE::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)       # sqrt of kinetic energy
    sqrtKE_u::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)   # interpolated on u-grid
    sqrtKE_v::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)   # interpolated on v-grid

    Bu::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη)         # bottom friction term u-component
    Bv::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη-1)         # bottom friction term v-component
end

"""Generator function for Bottomdrag VarCollection."""
function BottomdragVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return BottomdragVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""ArakawaHsu variables collected in a struct."""
@with_kw struct ArakawaHsuVars{T<:AbstractFloat}

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
    qα::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη-2)
    qβ::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-2)
    qγ::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-2)
    qδ::Array{T,2} = zeros(T,nx+2*haloη-2,ny+2*haloη-2)
end

"""Generator function for ArakawaHsu VarCollection."""
function ArakawaHsuVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return ArakawaHsuVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Laplace variables collected in a struct."""
@with_kw struct LaplaceVars{T<:AbstractFloat}

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

    Lu::Array{T,2} = zeros(T,nux+2*halo-2,nuy+2*halo-2)         # ∇²u
    Lv::Array{T,2} = zeros(T,nvx+2*halo-2,nvy+2*halo-2)         # ∇²v

    # Derivatives of Lu,Lv
    dLudx::Array{T,2} = zeros(T,nux+2*halo-3,nuy+2*halo-2)
    dLudy::Array{T,2} = zeros(T,nux+2*halo-2,nuy+2*halo-3)
    dLvdx::Array{T,2} = zeros(T,nvx+2*halo-3,nvy+2*halo-2)
    dLvdy::Array{T,2} = zeros(T,nvx+2*halo-2,nvy+2*halo-3)
end

"""Generator function for Laplace VarCollection."""
function LaplaceVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return LaplaceVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""Smagorinsky variables collected in a struct."""
@with_kw struct SmagorinskyVars{T<:AbstractFloat}

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

    DT::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)       # Tension squared (on the T-grid)
    DS::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)       # Shearing strain squared (on the T-grid)
    νSmag::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)    # Viscosity coefficient

    # Tension squared on the q-grid
    DS_q::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo)

    # Smagorinsky viscosity coefficient on the q-grid
    νSmag_q::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)

    # Entries of the Smagorinsky viscous tensor
    S12::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)
    S21::Array{T,2} = zeros(T,nx+2*haloη-1,ny+2*haloη-1)

    S11::Array{T,2} = zeros(T,nux+2*halo-3,nuy+2*halo-2)
    S22::Array{T,2} = zeros(T,nvx+2*halo-2,nvy+2*halo-3)

    # u- and v-components 1 and 2 of the biharmonic diffusion tendencies
    LLu1::Array{T,2} = zeros(T,nux+2*halo-4,nuy+2*halo-2)
    LLu2::Array{T,2} = zeros(T,nx+1,ny)

    LLv1::Array{T,2} = zeros(T,nx,ny+1)
    LLv2::Array{T,2} = zeros(T,nvx+2*halo-2,nvy+2*halo-4)
end

"""Generator function for Smagorinsky VarCollection."""
function SmagorinskyVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G

    return SmagorinskyVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη)
end

####################################################################

"""SemiLagrange variables collected in a struct."""
@with_kw struct SemiLagrangeVars{T<:AbstractFloat}

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

    xd::Array{T,2} = zeros(T,nx,ny)                         # departure points x-coord
    yd::Array{T,2} = zeros(T,nx,ny)                         # departure points y-coord

    um::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo)         # u-velocity temporal mid-point
    vm::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo)         # v-velocity temporal mid-point

    u_T::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)      # u-velocity interpolated on T-grid
    um_T::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)     # um interpolated on T-grid
    v_T::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)      # v-velocity interpolated on T-grid
    vm_T::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)     # vm interpolated on T-grid

    uinterp::Array{T,2} = zeros(T,nx,ny)                    # u interpolated on mid-point xd,yd
    vinterp::Array{T,2} = zeros(T,nx,ny)                    # v interpolated on mid-point xd,yd

    ssti::Array{T,2} = zeros(T,nx+2*halosstx,ny+2*halossty) # sst interpolated on departure points
    sst_ref::Array{T,2} = zeros(T,nx+2*halosstx,ny+2*halossty) # sst initial conditions for relaxation

    # compensated summation
    dsst_comp::Array{T,2} = zeros(T,nx+2*halosstx,ny+2*halossty)
end

"""Generator function for SemiLagrange VarCollection."""
function SemiLagrangeVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc = G
    @unpack halo,haloη = G
    @unpack halosstx,halossty = G

    return SemiLagrangeVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη,
                            halosstx=halosstx,halossty=halossty)
end

""" Variables that appear in Zanna-Bolton forcing term """
@with_kw struct ZBVars{T<:AbstractFloat}

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

    dudx::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)    # ∂u/∂x
    dudy::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo-1)    # ∂u/∂y
    dvdx::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo)    # ∂v/∂x
    dvdy::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)    # ∂v/∂y

    γ₀::Float64=0.3                       # coefficient in parameterization term

    # these are only utilized in a scheme where γ varies spacially
    γ::Array{T,2} = zeros(T,nx,ny)
    γ_u::Array{T,2} = zeros(T,nux,nuy)
    γ_v::Array{T,2} = zeros(T,nvx,nvy)

    Ker::Array{T,2} = zeros(3,3)    # convolutional kernal

    ζ::Array{T,2} = zeros(T,nqx,nqy)      # relative vorticity, cell corners
    ζsq::Array{T,2} = zeros(T,nqx,nqy)    # relative vorticity squared, cell corners

    D::Array{T,2} = zeros(T,nqx,nqy)      # shear deformation of flow field, cell corners
    Dsq::Array{T,2} = zeros(T,nqx,nqy)    # square of the tensor

    D_n::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo)
    D_nT::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη) 
    D_q::Array{T,2} = zeros(T,nqx,nqy)

    Dhat::Array{T,2} = zeros(T,nqx-1+2*haloη,nqy-1+2*haloη)     # stretch deformation of flow field, cell centers w/ halo
    Dhatsq::Array{T,2} = zeros(T,nqx-1+2*haloη,nqy-1+2*haloη)   # square of the tensor
    Dhatq::Array{T,2} = zeros(T,nqx,nqy)                  # tensor interpolated onto q-grid

    ζpDT::Array{T,2} = zeros(T,nx,ny)           # ζ^2 + D^2 interpolated to cell centers, not currently used
    ζsqT::Array{T,2} = zeros(T,nqx-1,nqy-1)     # ζ^2 interpolated to cell centers
    ζD::Array{T,2} = zeros(T,nqx,nqy)           # ζ ⋅ D, cell corners
    ζDT::Array{T,2} = zeros(T,nqx-1,nqy-1)      # ζ ⋅ D, placed on cell centers
    ζDhat::Array{T,2} = zeros(T,nqx,nqy)        # ζ ⋅ Dhat, cell corners
    
    trace::Array{T,2} = zeros(T,nx,ny)     # ξ^2 + D^2 + Dhat^2, cell centers

    ζD_filtered::Array{T,2} = zeros(T,nx,ny)      # ξD with filter applied
    ζDhat_filtered::Array{T,2} = zeros(T,nqx,nqy)   # ξDhat with filter applied
    trace_filtered::Array{T,2} = zeros(T,nx,ny)     # trace with filter applied

    dζDdx::Array{T,2} = zeros(T,nux,nuy)             # u-grid
    dζDhatdy::Array{T,2} = zeros(T,nux+halo,nuy)     # u-grid, initially with extra halo points
    dtracedx::Array{T,2} = zeros(T,nux,nuy)          # u-grid 

    S_u::Array{T,2} = zeros(T,nux,nuy)             # total forcing in x-direction

    dζDhatdx::Array{T,2} = zeros(T,nvx,nvy+halo)   # v-grid, initially with extra halo points
    dζDdy::Array{T,2} = zeros(T,nvx,nvy)           # v-grid
    dtracedy::Array{T,2} = zeros(T,nvx,nvy)        # v-grid

    S_v::Array{T,2} = zeros(T,nvx,nvy)             # total forcing in y-direction

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

    return ZBVars{T}(nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη,
                            halosstx=halosstx,halossty=halossty,Ker=Ker)
end

""" Variables that appear in NN forcing term """
@with_kw mutable struct NNVars{T<:AbstractFloat, OffDiagLayerType, DiagLayerType, OffDiagModelType, DiagModelType, OffDiagCompiledType, DiagCompiledType, DOffDiagCompiledType, DDiagCompiledType}

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

    γ₀::Float64=0.3                       # coefficient in parameterization term

    dudx::Array{T,2} = zeros(T,nux+2*halo-1,nuy+2*halo)    # ∂u/∂x
    dudy::Array{T,2} = zeros(T,nux+2*halo,nuy+2*halo-1)    # ∂u/∂y
    dvdx::Array{T,2} = zeros(T,nvx+2*halo-1,nvy+2*halo)    # ∂v/∂x
    dvdy::Array{T,2} = zeros(T,nvx+2*halo,nvy+2*halo-1)    # ∂v/∂y

    # weights_corner is applied to ζ and D to form the diagonal terms of S,
    # and weights_center is applied to ζ and D̃ to form the off-diagonal terms
    # of S. Initially this will just be a single layer to see if we can get the model
    # with the neural net running
    # the initial weight values might need to change, for now I'm setting them to zero
    # offdiag_outdim::Int
    # offdiag_indim::Int

    # diag_outdim::Int
    # diag_indim::Int

    ζ::Array{T,2} = zeros(T,nqx,nqy)      # relative vorticity, cell corners

    D::Array{T,2} = zeros(T,nqx,nqy)      # shear deformation of flow field, cell corners

    Dhat::Array{T,2} = zeros(T,nx+2*haloη,ny+2*haloη)     # stretch deformation of flow field, cell centers w/ halo

    ζT::Array{T,2} = zeros(T,nx,ny)         # ζ interpolated to cell centers
    DT::Array{T,2} = zeros(T,nx,ny)         # D, interpolated on cell centers
    ζDhat::Array{T,2} = zeros(T,nqx,nqy)    # ζ ⋅ Dhat, cell corners

    T11::Array{T,2} = zeros(T,nx,ny)
    T12::Array{T,2} = zeros(T,nqx,nqy)
    T22::Array{T,2} = zeros(T,nx,ny)

    dT11dx::Array{T,2} = zeros(T,nux,nuy)    # derivative of T11 in the x-direction, u-grid
    dT12dy::Array{T,2} = zeros(T,nux+halo,nuy)    # derivative of T12 in the y-direction, u-grid
    dT12dx::Array{T,2} = zeros(T,nvx,nvy+halo)    # derivative of T12 in the x-direction, v-grid
    dT22dy::Array{T,2} = zeros(T,nvx,nvy)    # derivative of T22 in the y-direction, v-grid

    S_u::Array{T,2} = zeros(T,nux,nuy)             # total forcing in x-direction
    S_v::Array{T,2} = zeros(T,nvx,nvy)             # total forcing in y-direction

    offdiag_layers::OffDiagLayerType
    diag_layers::DiagLayerType

    model_offdiag::OffDiagModelType
    model_diag::DiagModelType

    compiled_offdiag::OffDiagCompiledType
    compiled_diag::DiagCompiledType

    compiled_doffdiag::DOffDiagCompiledType
    compiled_ddiag::DDiagCompiledType
end

function nn_apply(result, layers, input, ps, st)
    res2 = Lux.apply(layers, input, ps, st)[1]
    copyto!(result, Base.reshape(res2, size(result)))
    nothing
end

using Enzyme

function grad_apply(dres, dps, layers, input, dinput, ps, st)
    res = similar(dres)
    Enzyme.autodiff(Reverse, Const(nn_apply), Duplicated(res, dres), Const(layers), Duplicated(input, dinput), Duplicated(ps, dps), Const(st))
    return nothing
end

"""Generator function for NN momentum terms."""
# one layer
function NNVars{T}(G::Grid) where {T<:AbstractFloat}

    @unpack nx,ny,bc,Δ= G
    @unpack halo,haloη = G
    @unpack halosstx,halossty = G

    nqx = if (bc == "periodic") nx else nx+1 end      # q-grid in x-direction
    nqy = ny+1                                        # q-grid in y-direction

    offdiag_dims = [22, 25, 20, 20, 20, 20, 20, 1]
    diag_dims = [17, 20, 20, 20, 20, 20, 20, 2]

    offdiag_layers = Lux.Chain(
        (
            Lux.Dense(offdiag_dims[i] => offdiag_dims[i+1], (i == (length(offdiag_dims)-1) ? identity : gelu))
            for i in 1:(length(offdiag_dims)-1)
        )...
    )
    
    diag_layers = Lux.Chain(
        (
            Lux.Dense(diag_dims[i] => diag_dims[i+1], (i == (length(diag_dims)-1) ? identity : gelu))
            for i in 1:(length(diag_dims)-1)
        )...
    )

    model_offdiag = Lux.setup(Random.default_rng(), offdiag_layers)
    model_diag = Lux.setup(Random.default_rng(), diag_layers)

    offdiag_input = Reactant.to_rarray(Array{T}(undef, 9+9+4, nqx, nqy))
    diag_input = Reactant.to_rarray(Array{T}(undef, 9+4+4, nx, ny))

    offdiag_dinput = Reactant.to_rarray(Array{T}(undef, 9+9+4, nqx, nqy))
    diag_dinput = Reactant.to_rarray(Array{T}(undef, 9+4+4, nx, ny))

    d_offdiag_res = Reactant.to_rarray(Array{T}(undef, 1, nqx, nqy))
    d_diag_res = Reactant.to_rarray(Array{T}(undef, 2, nx, ny))

    use_reactant = false

    if use_reactant
        model_offdiag = Reactant.to_rarray(model_offdiag)
    end
    if use_reactant
        model_diag = Reactant.to_rarray(model_diag)
    end

    if use_reactant
        compiled_offdiag = Reactant.@compile Lux.apply(offdiag_layers, offdiag_input, model_offdiag[1], model_offdiag[2])
        compiled_diag = Reactant.@compile Lux.apply(diag_layers, diag_input, model_diag[1], model_diag[2])

        compiled_doffdiag = Reactant.@compile grad_apply(d_offdiag_res, deepcopy(model_offdiag[1]), offdiag_layers, offdiag_input, offdiag_dinput, model_offdiag[1], model_offdiag[2])
        compiled_ddiag = Reactant.@compile grad_apply(d_diag_res, deepcopy(model_diag[1]), diag_layers, diag_input, diag_dinput, model_diag[1], model_diag[2])
    else
        compiled_offdiag = nothing
        compiled_diag = nothing
        compiled_doffdiag = nothing
        compiled_ddiag = nothing
    end


    return NNVars{T, typeof(offdiag_layers), typeof(diag_layers), typeof(model_offdiag), typeof(model_diag), typeof(compiled_offdiag), typeof(compiled_diag), typeof(compiled_doffdiag), typeof(compiled_ddiag)}(; nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη,
                    halosstx=halosstx,halossty=halossty, offdiag_layers, diag_layers, model_offdiag, model_diag, compiled_offdiag, compiled_diag, compiled_doffdiag, compiled_ddiag
    )
end

# four layers
# function NNVars{T}(G::Grid) where {T<:AbstractFloat}

#     @unpack nx,ny,bc,Δ= G
#     @unpack halo,haloη = G
#     @unpack halosstx,halossty = G

#     nqx = if (bc == "periodic") nx else nx+1 end      # q-grid in x-direction
#     nqy = ny+1                                        # q-grid in y-direction

#     # am going to try increasing to more than one layer
#     offdiag_indim = 22
#     offdiag_outdim = 10
#     offdiag_indim1 = 10
#     offdiag_outdim1 = 10
#     offdiag_indim2 = 10
#     offdiag_outdim2 = 10
#     offdiag_indim_final = 10
#     offdiag_outdim_final = 1

#     diag_indim = 17
#     diag_outdim = 10
#     diag_indim1 = 10
#     diag_outdim1 = 10
#     diag_indim2 = 10
#     diag_outdim2 = 10
#     diag_indim_final = 10
#     diag_outdim_final = 2

#     offdiag_layers = Lux.Chain(Lux.Dense(offdiag_indim => offdiag_outdim, sigmoid),
#         Lux.Dense(offdiag_indim1 => offdiag_outdim1),
#         Lux.Dense(offdiag_indim2 => offdiag_outdim2),
#         Lux.Dense(offdiag_indim_final => offdiag_outdim_final, sigmoid)
#     )
#     diag_layers = Lux.Chain(Lux.Dense(diag_indim => diag_outdim, sigmoid),
#         Lux.Dense(diag_indim1 => diag_outdim1),
#         Lux.Dense(diag_indim2 => diag_outdim2),
#         Lux.Dense(diag_indim_final => diag_outdim_final, sigmoid)
#     )

#     model_offdiag = Lux.setup(Random.default_rng(), offdiag_layers)
#     model_diag = Lux.setup(Random.default_rng(), diag_layers)

#     offdiag_input = Reactant.to_rarray(Array{T}(undef, 9+9+4, nqx, nqy))
#     diag_input = Reactant.to_rarray(Array{T}(undef, 9+4+4, nx, ny))

#     offdiag_dinput = Reactant.to_rarray(Array{T}(undef, 9+9+4, nqx, nqy))
#     diag_dinput = Reactant.to_rarray(Array{T}(undef, 9+4+4, nx, ny))

#     d_offdiag_res = Reactant.to_rarray(Array{T}(undef, 1, nqx, nqy))
#     d_diag_res = Reactant.to_rarray(Array{T}(undef, 2, nx, ny))

#     use_reactant = false

#     if use_reactant
#         model_offdiag = Reactant.to_rarray(model_offdiag)
#     end
#     if use_reactant
#         model_diag = Reactant.to_rarray(model_diag)
#     end

#     if use_reactant
#         compiled_offdiag = Reactant.@compile Lux.apply(offdiag_layers, offdiag_input, model_offdiag[1], model_offdiag[2])
#         compiled_diag = Reactant.@compile Lux.apply(diag_layers, diag_input, model_diag[1], model_diag[2])

#         compiled_doffdiag = Reactant.@compile grad_apply(d_offdiag_res, deepcopy(model_offdiag[1]), offdiag_layers, offdiag_input, offdiag_dinput, model_offdiag[1], model_offdiag[2])
#         compiled_ddiag = Reactant.@compile grad_apply(d_diag_res, deepcopy(model_diag[1]), diag_layers, diag_input, diag_dinput, model_diag[1], model_diag[2])
#     else
#         compiled_offdiag = nothing
#         compiled_diag = nothing
#         compiled_doffdiag = nothing
#         compiled_ddiag = nothing
#     end


#     return NNVars{T, typeof(offdiag_layers), typeof(diag_layers), typeof(model_offdiag), typeof(model_diag), typeof(compiled_offdiag), typeof(compiled_diag), typeof(compiled_doffdiag), typeof(compiled_ddiag)}(; nx=nx,ny=ny,bc=bc,halo=halo,haloη=haloη,
#                     halosstx=halosstx,halossty=halossty, offdiag_layers, diag_layers, model_offdiag, model_diag, compiled_offdiag, compiled_diag, compiled_doffdiag, compiled_ddiag
#     )
# end

"""Preallocate the diagnostic variables and return them as matrices in structs."""
function preallocate(   ::Type{T},
                        ::Type{Tprog},
                        G::Grid) where {T<:AbstractFloat,Tprog<:AbstractFloat}

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
