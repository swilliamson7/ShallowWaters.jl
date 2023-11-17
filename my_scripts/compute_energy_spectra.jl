function compute_energy_spectra(path_to_u_velocity::String, path_to_v_velocity::String)

    u = ncread(path_to_u_velocity)
    v = ncread(path_to_v_velocity)

    u_velocity_transformed = zeros(size(u))
    v_velocity_transformed = zeros(size(v))

    _, _, t = size(u)

    for j = 1:t 

        u_velocity_transformed[:,:,j] .= fft(u[:, :, j])
        v_velocity_transformed[:,:,j] .= fft(v[:, :, j])
    end
end