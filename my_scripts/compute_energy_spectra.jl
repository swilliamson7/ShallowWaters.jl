using DSP, FFTW
using NetCDF, Parameters, Printf, Dates, Interpolations
using Plots

function plot_energy_spectra()

    u_noforcing = ncread("./data_files_gamma0.3/128_spinup_noforcing/u.nc", "u")
    v_noforcing = ncread("./data_files_gamma0.3/128_spinup_noforcing/v.nc", "v")

    u_withparam_nofilter = ncread("./data_files_gamma0.3/128_spinup_wforcing_dissipation_nofilter/u.nc", "u")
    v_withparam_nofilter = ncread("./data_files_gamma0.3/128_spinup_wforcing_dissipation_nofilter/v.nc", "v")

    u_withparam_withfilter1 = ncread("./data_files_gamma0.3/128_spinup_wforcing_dissipation_wfilter_1pass/u.nc", "u")
    v_withparam_withfilter1 = ncread("./data_files_gamma0.3/128_spinup_wforcing_dissipation_wfilter_1pass/v.nc", "v")

    u_withparam_withfilter4 = ncread("./data_files_gamma0.3/128_spinup_wforcing_dissipation_wfilter_4pass/u.nc", "u")
    v_withparam_withfilter4 = ncread("./data_files_gamma0.3/128_spinup_wforcing_dissipation_wfilter_4pass/v.nc", "v")

    u_hr = ncread("./data_files_gamma0.3/512_spinup/u.nc", "u")
    v_hr = ncread("./data_files_gamma0.3/512_spinup/v.nc", "v")

    u1 = u_noforcing[:, :, end]
    v1 = v_noforcing[:, :, end]

    u1_hat = periodogram(u1; nfft=nextfastfft(size(u1)), fs=1, radialsum=true)
    v1_hat = periodogram(v1; nfft=nextfastfft(size(v1)), fs=1, radialsum=true)

    u2 = u_withparam_nofilter[:, :, end]
    v2 = v_withparam_nofilter[:, :, end]

    u2_hat = periodogram(u2; nfft=nextfastfft(size(u2)), fs=1, radialsum=true)
    v2_hat = periodogram(v2; nfft=nextfastfft(size(v2)), fs=1, radialsum=true)

    u3 = u_withparam_withfilter1[:, :, end]
    v3 = v_withparam_withfilter1[:, :, end]

    u3_hat = periodogram(u3; nfft=nextfastfft(size(u3)), fs=1, radialsum=true)
    v3_hat = periodogram(v3; nfft=nextfastfft(size(v3)), fs=1, radialsum=true)

    u4 = u_hr[:, :, end]
    v4 = v_hr[:, :, end]

    u4_hat = periodogram(u4; nfft=nextfastfft(size(u4)), fs=1, radialsum=true)
    v4_hat = periodogram(v4; nfft=nextfastfft(size(v4)), fs=1, radialsum=true)

    u5 = u_withparam_withfilter4[:, :, end]
    v5 = v_withparam_withfilter4[:, :, end]

    u5_hat = periodogram(u5; nfft=nextfastfft(size(u5)), fs=1, radialsum=true)
    v5_hat = periodogram(v5; nfft=nextfastfft(size(v5)), fs=1, radialsum=true)

    plot(u1_hat.freq[2:end], u1_hat.power[2:end] .+ v1_hat.power[2:end], xaxis=:log, yaxis=:log, label="30km res.")

    plot!(u2_hat.freq[2:end], u2_hat.power[2:end] .+ v2_hat.power[2:end], xaxis=:log, yaxis=:log, label="30km res., with closure")

    plot!(u3_hat.freq[2:end], u3_hat.power[2:end] .+ v3_hat.power[2:end], xaxis=:log, yaxis=:log, label="30km res. with 1 pass filtered closure")

    plot!(u5_hat.freq[2:end], u5_hat.power[2:end] .+ u5_hat.power[2:end], xaxis=:log, yaxis=:log, label="30km res. with 4 pass filtered closure")

    plot!(u4_hat.freq[2:end], u4_hat.power[2:end] .+ v4_hat.power[2:end], xaxis=:log, yaxis=:log, label="7.5km res.")

    xlabel!("k")
    ylabel!("E(k)")


end

