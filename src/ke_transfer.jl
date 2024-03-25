using FFTW

function compute_ke_transfer(S)

    S_u = S.Diag.ZBVars.S_u
    S_v = S.Diag.ZBVars.S_v

    P = PrognosticVars{Float32}(remove_halo(S.Prog.u,S.Prog.v,S.Prog.η,S.Prog.sst,S)...)

    tau_1 = conj(fft(P.u)) .* fft(S_u)
    tau_2 = conj(fft(P.v)) .* fft(S_v)

end