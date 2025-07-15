# Implements a Gaussian filter to coarse-grain high resolution prognostic variables to be at the resolution determined by S_lr
# Needs the states without a halo, will not work for haloed prognostic variables

function coarse_grain(u_hr, v_hr, η_hr, nx_hr, S_lr)

    Prog_lr = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(
        S_lr.Prog.u,
        S_lr.Prog.v,
        S_lr.Prog.η,
        S_lr.Prog.sst,
        S_lr)...
    )

    mu, nu = size(Prog_lr.u)
    mu_hr, nu_hr = size(u_hr)
    mv, nv = size(Prog_lr.v)
    mv_hr, nv_hr = size(v_hr)
    meta, neta = size(Prog_lr.η)
    meta_hr, neta_hr = size(η_hr)

    ū_hr = zeros(mu, nu)
    v̄_hr = zeros(mv, nv)
    η̅_hr = zeros(meta, neta)

    # needs to be adjusted for any grid that isn't square
    x_lr = 0:S_lr.grid.Δ:S_lr.parameters.Lx
    y_lr = 0:S_lr.grid.Δ:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    Δ_hr = S_lr.parameters.Lx / nx_hr
    x_hr = 0:Δ_hr:S_lr.parameters.Lx
    y_hr = 0:Δ_hr:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    coeffu = zeros(mu, nu)
    coeffv = zeros(mv, nv)
    coeffeta = zeros(meta,neta)

    den = 2 * 30e3^2

    @inbounds for k ∈ 1:mu, j ∈ 1:nu

        @inbounds for k2 ∈ 1:mu_hr
            for j2 ∈ 1:nu_hr

                ex = exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/den)
                coeffu[k,j] += ex
                ū_hr[k,j] += u_hr[k2,j2] * ex

            end
        end

        ū_hr[k,j] = (1 / coeffu[k,j]) * ū_hr[k,j]

    end

    @inbounds for j ∈ 1:nv, k ∈ 1:mv

        @inbounds for j2 ∈ 1:nv_hr
            for k2 ∈ 1:mv_hr

                ex = exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/den)
                coeffv[k,j] += ex
                v̄_hr[k,j] += v_hr[k2,j2] * ex

            end
        end

        v̄_hr[k,j] = (1 / coeffv[k,j]) * v̄_hr[k,j]

    end

    @inbounds for j ∈ 1:neta, k ∈ 1:meta

        @inbounds for j2 ∈ 1:neta_hr
            for k2 ∈ 1:meta_hr

                ex = exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/den)
                coeffeta[k,j] += ex
                η̅_hr[k,j] += η_hr[k2,j2] * ex

            end
        end

        η̅_hr[k,j] = (1 / coeffeta[k,j]) * η̅_hr[k,j]

    end

    return ū_hr, v̄_hr, η̅_hr

end

function coarse_grain_u(u_hr, nx_hr, S_lr)

    Prog_lr = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(
        S_lr.Prog.u,
        S_lr.Prog.v,
        S_lr.Prog.η,
        S_lr.Prog.sst,
        S_lr)...
    )

    mu, nu = size(Prog_lr.u)
    mu_hr, nu_hr = size(u_hr)
    ū_hr = zeros(mu, nu)

    # needs to be adjusted for any grid that isn't square
    x_lr = 0:S_lr.grid.Δ:S_lr.parameters.Lx
    y_lr = 0:S_lr.grid.Δ:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    Δ_hr = S_lr.parameters.Lx / nx_hr
    x_hr = 0:Δ_hr:S_lr.parameters.Lx
    y_hr = 0:Δ_hr:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    coeffu = zeros(mu, nu)
    den = 2 * 30e3^2

    @inbounds for k ∈ 1:mu, j ∈ 1:nu

        @inbounds for k2 ∈ 1:mu_hr
            for j2 ∈ 1:nu_hr

                ex = exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/den)
                coeffu[k,j] += ex
                ū_hr[k,j] += u_hr[k2,j2] * ex

            end
        end

        ū_hr[k,j] = (1 / coeffu[k,j]) * ū_hr[k,j]

    end

    return ū_hr

end

function coarse_grain_v(v_hr, nx_hr, S_lr)

    Prog_lr = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(
        S_lr.Prog.u,
        S_lr.Prog.v,
        S_lr.Prog.η,
        S_lr.Prog.sst,
        S_lr)...
    )

    mv, nv = size(Prog_lr.v)
    mv_hr, nv_hr = size(v_hr)

    v̄_hr = zeros(mv, nv)

    # needs to be adjusted for any grid that isn't square
    x_lr = 0:S_lr.grid.Δ:S_lr.parameters.Lx
    y_lr = 0:S_lr.grid.Δ:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    Δ_hr = S_lr.parameters.Lx / nx_hr
    x_hr = 0:Δ_hr:S_lr.parameters.Lx
    y_hr = 0:Δ_hr:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    coeffv = zeros(mv, nv)

    den = 2 * 30e3^2

    @inbounds for j ∈ 1:nv, k ∈ 1:mv

        @inbounds for j2 ∈ 1:nv_hr
            for k2 ∈ 1:mv_hr

                ex = exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/den)
                coeffv[k,j] += ex
                v̄_hr[k,j] += v_hr[k2,j2] * ex

            end
        end

        v̄_hr[k,j] = (1 / coeffv[k,j]) * v̄_hr[k,j]

    end

    return v̄_hr

end

function coarse_grain_eta(eta_hr, nx_hr, S_lr)

    Prog_lr = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(
        S_lr.Prog.u,
        S_lr.Prog.v,
        S_lr.Prog.η,
        S_lr.Prog.sst,
        S_lr)...
    )

    meta, neta = size(Prog_lr.η)
    meta_hr, neta_hr = size(eta_hr)

    η̅_hr = zeros(meta, neta)

    # needs to be adjusted for any grid that isn't square
    x_lr = 0:S_lr.grid.Δ:S_lr.parameters.Lx
    y_lr = 0:S_lr.grid.Δ:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    Δ_hr = S_lr.parameters.Lx / nx_hr
    x_hr = 0:Δ_hr:S_lr.parameters.Lx
    y_hr = 0:Δ_hr:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    coeffeta = zeros(meta, neta)

    den = 2 * 30e3^2

    @inbounds for j ∈ 1:neta, k ∈ 1:meta

        @inbounds for j2 ∈ 1:neta_hr
            for k2 ∈ 1:meta_hr

                ex = exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/den)
                coeffeta[k,j] += ex
                η̅_hr[k,j] += η_hr[k2,j2] * ex

            end
        end

        η̅_hr[k,j] = (1 / coeffeta[k,j]) * η̅_hr[k,j]

    end

    return η̅_hr

end