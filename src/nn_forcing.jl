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

    T_xx = zeros(127,128)
    T_xy = zeros(127,128)
    T_yy = zeros(128,127)
    T_yx = zeros(128,127)

end

function NN_momentum_centers(u, v, S, Diag, weights)



end

function NN_momentum(u, v, S, Diag, weights1, weights2)