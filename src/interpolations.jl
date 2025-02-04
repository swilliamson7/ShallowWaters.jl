"""Linear interpolation of a variable u in the x-direction.
m,n = size(ux) must be m+1,n = size(u)."""
function Ix!(ux,u)
    m, n = size(ux)
    @boundscheck (m+1,n) == size(u) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            ux[i,j] = 0.5*(u[i+1,j] + u[i,j])
        end
    end
end

""" Linear interpolation a variable u in the y-direction.
    m,n = size(uy) must be m,n+1 = size(u)."""
function Iy!(uy,u)
    m,n = size(uy)
    @boundscheck (m,n+1) == size(u) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            uy[i,j] = 0.5*(u[i,j+1] + u[i,j])
        end
    end
end

""" Bilinear interpolation a variable u in x and y-direction.
m,n = size(uxy) must be m+1,n+1 = size(u). """
function Ixy!(uxy,u)
    m,n = size(uxy)
    @boundscheck (m+1,n+1) == size(u) || throw(BoundsError())

    @inbounds for j in 1:n, i in 1:m
        uxy[i,j] = 0.25*(u[i,j] + u[i+1,j]) + 
                0.25*(u[i,j+1] + u[i+1,j+1])
    end
end

function Ix(u)
    m,n = size(u)
    ux = typeof(u)(undef,m-1,n)


    @inbounds for j ∈ 1:n
        for i ∈ 1:m-1
            ux[i,j] = 0.5*(u[i+1,j] + u[i,j])
        end
    end

    return ux
end

function Iy(u)
    m,n = size(u)
    uy = typeof(u)(undef,m,n-1)

    @inbounds for j ∈ 1:n-1
        for i ∈ 1:m
            uy[i,j] = 0.5*(u[i,j+1] + u[i,j])
        end
    end

    return uy
end
