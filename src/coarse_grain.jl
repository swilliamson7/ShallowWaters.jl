"""
Will take in a high resolution u, v, and η (velocity fields and surface displacement, respectively) and 
coarse-grain them to return u_bar, v_bar, and η_bar. The coarse graining will be done with a 
Gaussian low pass filtering operation, mimicking the definition given in Bolton and Zanna (2019) Eq. (4)
"""
function coarse_grain(u_hr, v_hr, η_hr, nx_hr, ny_hr, S_lr)

    σ = 30

    u_lr = S_lr.Prog.u
    v_lr = S_lr.Prog.v
    η_lr = S_lr.Prog.η

    mu, nu = size(u_lr)
    mu_hr, nu_hr = size(u_hr)
    mv, nv = size(v_lr)
    mv_hr, nv_hr = size(v_hr)
    meta, neta = size(η_lr)
    meta_hr, neta_hr = size(η_hr)

    ū_hr = zeros(mu, nu)
    v̄_hr = zeros(mv, nv)
    η̅_hr = zeros(meta, neta)

    # needs to be adjusted for any grid that isn't square
    x_lr = 0:S_lr.grid.Δ:S_lr.parameters.Lx
    y_lr = 0:S_lr.grid.Δ:(S_lr.parameter.Lx/S_lr.parameters.L_ratio)

    Δ_hr = S_lr.parameters.Lx / nx_hr

    x_hr = 0:Δ_hr:S_lr.parameters.Lx
    y_hr = 0:Δ_hr:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    coeff = 1 / (2 * π * σ^2)

    for j ∈ nu, k ∈ mu

        for j2 ∈ nu_hr
            for k2 ∈ mu_hr

                ū_hr[k,j] = ū_hr[k,j] + u_hr[k2,j2] * exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/(2 * σ^2))

            end
        end

        ū_hr = coeff * ū_hr

    end

    for j ∈ 1:nv, k ∈ 1:mv

        for j2 ∈ 1:nv_hr
            for k2 ∈ 1:mv_hr

                v̄_hr[k,j] = v̄_hr[k,j] + v_hr[k2,j2] * exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/(2 * σ^2))

            end
        end

        v̄_hr = coeff * v̄_hr

    end

    for j ∈ 1:neta, k ∈ 1:meta

        for j2 ∈ 1:neta_hr
            for k2 ∈ 1:meta_hr

                η̅_hr[k,j] = η̅_hr[k,j] + η_hr[k2,j2] * exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/(2 * σ^2))

            end
        end

        η̅_hr = coeff * η̅_hr

    end

    return ū_hr, v̄_hr, η̅_hr

end

function coarse_grain_eta(η_hr, S_lr)

    # this is the standard deviation of the Gaussian filter, "...determines the 
    # length scale at which information is removed
    σ = 30 * 1e3

    meta = S_lr.grid.ny
    neta = S_lr.grid.nx
    meta_hr, neta_hr = size(η_hr)

    η̅_hr = zeros(meta, neta)

    # needs to be adjusted for any grid that isn't square
    x_lr = 0:(S_lr.grid.Δ ):(S_lr.parameters.Lx )
    y_lr = 0:(S_lr.grid.Δ ):(S_lr.parameters.Lx/(S_lr.parameters.L_ratio ))

    Δ_hr = S_lr.parameters.Lx / (1024)

    x_hr = 0:Δ_hr:(S_lr.parameters.Lx )
    y_hr = 0:Δ_hr:(S_lr.parameters.Lx/(S_lr.parameters.L_ratio ))

    coeff = 1 / (2 * π * σ^2)

    @inbounds for j ∈ 1:neta, k ∈ 1:meta

        @inbounds for j2 ∈ 1:neta_hr
            @inbounds for k2 ∈ 1:meta_hr

                η̅_hr[k,j] = η̅_hr[k,j] + η_hr[k2,j2] * exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/(2 * σ^2))

            end
        end

    end

    η̅_hr = coeff * η̅_hr

    return η̅_hr

end