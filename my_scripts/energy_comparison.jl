using JLD2, Plots

energy1 = load_object("./data_files/128_spinup_wforcing_dissipation_nofilter/energy_during_spinup_128_withzb_dissipation_wofilter_10years_110823.jld2")
energy2 = load_object("./data_files/128_spinup_wforcing_dissipation_wfilter_1pass/energy_during_spinup_128_withzb_dissipation_wfilter_1pass_10years_110823.jld2")
energy3 = load_object("./data_files/128_spinup_wforcing_dissipation_wfilter_4pass/energy_during_spinup_128_withzb_dissipation_wfilter_4pass_10years_110823.jld2")
energy4 = load_object("./data_files/128_spinup_wforcing_momentum_filtered_1pass/energy_during_spinup_128_withzb_momentum_wfilter_1pass_10years_110823.jld2")
energy4_1 = load_object("./data_files/128_spinup_wforcing_momentum_wfilter_4pass/energy_during_spinup_128_withzb_momentum_wfilter_4pass_10years_110823.jld2")
energy5 = load_object("./data_files/128_spinup_wforcing_momentum_nofilter/energy_during_spinup_128_withzb_momentum_wofilter_10years_110823.jld2")
energy6 = load_object("./data_files/128_spinup_noforcing/energy_during_spinup_128_10years_110823.jld2")
energy7 = load_object("./data_files/512_spinup/energy_during_spinup_512_10years_110823.jld2")

plot(0:1:10*3650, energy1, label="Dissipation, no filter",xlabel="Days",ylabel="Spatially averaged energy")
plot!(0:1:10*3650, energy2, label="Dissipation, 1 pass filter",xlabel="Days",ylabel="Spatially averaged energy")
plot!(0:1:10*3650, energy3, label="Dissipation, 4 pass filter",xlabel="Days",ylabel="Spatially averaged energy")
plot!(0:1:10*3650, energy4, label="Momentum, 1 pass filter",xlabel="Days",ylabel="Spatially averaged energy")
plot!(0:1:10*3650, energy4_1, label="Momentum, 4 pass filter",xlabel="Days",ylabel="Spatially averaged energy")
plot!(0:1:10*3650, energy5, label="Momentum, no filter",xlabel="Days",ylabel="Spatially averaged energy")
plot!(0:1:10*3650, energy6, label="30km resolution, no parameterization",xlabel="Days",ylabel="Spatially averaged energy")
plot!(0::10*3650, energy7, label="7.5km resolution",xlabel="Days",ylabel="Spatially averaged energy")

savefig("energy_comparison_all.png")