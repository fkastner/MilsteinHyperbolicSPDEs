## numerical simulation of a stochastic Schrödinger equation
## for the paper 'Milstein-type Schemes for Hyperbolic SPDEs'
## original authors: 
##     Felix Kastner, University of Hamburg
##     Katharina Klioba, Delft University of Technology
## MIT License, Copyright (c) 2026

using Random: seed!
using ProgressMeter
using DelimitedFiles
using CairoMakie

include("tools_common.jl")
include("tools_fourier.jl")

seed!(43)

## number of samples
N_samples = 100

## space discretisation (via spectral Galerkin)
M = 2^10 # number of Fourier modes = frequencies -M/2+1:M/2
setupFourierTools(M) # 'xx' are space coordinates, 'kk' are frequencies

## initial value in frequency space
Fu_0 = Complex.(1 ./ (1 .+ abs.(kk) .^ 6))

## eigenvalues for the noise
β = 5.1
Lambda = (1 ./ (1 .+ abs.(kk) .^ β))
sqrtLambda = sqrt.(Lambda)

## convolution kernel
η = periodicbump.(xx) # smooth bump function centered at zero with compact support
Fη = dft(η)

## nonlinearity (Nemytskii kernel)
ϕ(z) = z / (1 + abs(z)^2)

## precompute (part of) Milstein term
Fsum_eta_g = 1 / sqrt(2 * pi) * vec([zeros(1, M ÷ 2); Lambda[M÷4+1:M÷2+M÷4]'])

## time discretisation
t_0, t_end = 0, 0.5
dt_exact = 2.0^(-14) # time step for reference solution
dt_num = 2.0 .^ (-5:-1:-9) # vector of different time steps
N_dt = length(dt_num)
Nt_exact = floor(Int, (t_end - t_0) / dt_exact)

function euler(R, ts, FΔW, Fu0)
    M = length(Fu0)
    dt = step(ts)
    FU = similar(Fu0, M, length(ts))
    FU[:, 1] = Fu0
    drift_term = similar(Fu0)
    noise_term = similar(Fu0)
    tmp_conv = zeros(ComplexF64, 2M - 1) # temporary workspace for the convolution
    for (i, t) in enumerate(ts[2:end])
        drift_term .= -1im * dt * Fη .* applyFourierFunc(ϕ, FU[:, i]) # F(u) = -i η∗ϕ(u)
        noise_term .= -1im * conv_trunc!(tmp_conv, FΔW[:, i], FU[:, i]) # G(u) = -iu
        FU[:, i+1] .= R .* (FU[:, i] .+ drift_term .+ noise_term)
    end
    FU
end
function milstein(R, ts, FΔW, Fu0)
    M = length(Fu0)
    dt = step(ts)
    FU = similar(Fu0, M, length(ts))
    FU[:, 1] = Fu0
    drift_term = similar(Fu0)
    noise_term = similar(Fu0)
    tmp_conv = zeros(ComplexF64, 2M - 1) # temporary workspace for the convolution
    for (i, t) in enumerate(ts[2:end])
        drift_term .= -1im * dt * Fη .* applyFourierFunc(ϕ, FU[:, i]) # F(u) = -i η∗ϕ(u)
        noise_term .= -1im * conv_trunc!(tmp_conv, FΔW[:, i], FU[:, i]) # G(u) = -iu
        FΔW_sq = conv_trunc!(tmp_conv, FΔW[:, i], FΔW[:, i])
        noise_term2 = -1 / 2 * conv_trunc!(tmp_conv, FU[:, i], FΔW_sq - dt * Fsum_eta_g)
        FU[:, i+1] .= R .* (FU[:, i] .+ drift_term .+ noise_term .+ noise_term2)
    end
    FU
end

time_disc = [euler, milstein]
time_disc_labels = ["E", "M"]

## semigroup approximation methods to consider
semigroup_approx = [R_IE, R_CN, R_EE]
semigroup_approx_labels = ["LI", "CN", "EX"]

## construct schemes from time discretizations × semigroup approximations
schemes = tuple.(semigroup_approx, permutedims(time_disc))
scheme_labels = semigroup_approx_labels .* permutedims(time_disc_labels)
N_schemes = length(schemes)
@assert N_schemes == length(scheme_labels)

## preallocate
FU_exact = zeros(ComplexF64, M, Nt_exact + 1) # (Fourier coeff. of) exact solution via exponential Euler with small time step
UniformErrorSamples = zeros(N_schemes, N_dt, N_samples) # schemes, N_dt different stepsizes, N_samples samples

# main loop: simulate pathwise
@info "Nonlinear Schroedinger with Convolution" β t_end dt_exact M N_samples

@showprogress showspeed = true desc = "Computing..." for m = 1:N_samples
    ## simulate noise
    FΔW = sqrtLambda .* sqrt(dt_exact) .* randn(M, Nt_exact) # Fourier coeff. of Q-BM

    ## compute reference solution
    ts = t_0:dt_exact:t_end
    Rk = R_EE(dt_exact, M)
    FU_exact .= euler(Rk, ts, FΔW, Fu_0)

    ## compute approximations
    for (j, dt) in enumerate(dt_num)
        ts_coarse = t_0:dt:t_end
        Nt = length(ts_coarse) - 1
        ratio = Int(dt / dt_exact)
        FΔW_coarse = reshape(sum(reshape(FΔW', ratio, Nt, M), dims=1), Nt, M)'

        for (i, (R, time_discretisation)) in enumerate(schemes)
            Rk = R(dt, M)
            FU = time_discretisation(Rk, ts_coarse, FΔW_coarse, Fu_0)
            UniformErrorSamples[i, j, m] = uniform_error(FU, FU_exact[:, 1:ratio:Nt_exact+1])
        end
    end
end # m 

## estimate numerical convergence rates
UniformErrors = L2norm.(eachslice(UniformErrorSamples, dims=(1, 2)))
averageConvRates = avgRate.([dt_num], eachslice(UniformErrors, dims=1))

@info round.(avgRate.([dt_num[1:end-2]], eachslice(UniformErrors[:, 1:end-2], dims=1)), digits=2)

## save data to disk
data = ["Stepsize" scheme_labels...;
    dt_num round.(UniformErrors, digits=4)']
path = mkpath("data")
writedlm(joinpath(path, "NLSE_bumpconv_β_$(Int(10β))_T_$(t_end)_dt0_$(Int(log2(dt_exact)))_M_$(Int(log2(M)))_samples_$(N_samples).csv"), data)

## convergence plot
fig = Figure()
ax = Axis(fig[1, 1],
    title="NLSE: β = $(β), dt₀ = 2^($(log2(dt_exact))), M = 2^$(log2(M))",
    xlabel="Stepsize",
    ylabel="Pathwise Uniform Error",
    xscale=log2,
    yscale=log2
)

lines!(ax, dt_num, dt_num .^ (0.25) * UniformErrors[2, 1] / dt_num[1]^(0.25) * 1.2, color=:black, linestyle=:dot, label="Slope 1/4")
lines!(ax, dt_num, dt_num .^ (0.5) * UniformErrors[1, 1] / dt_num[1]^(0.5) * 0.9, color=:black, linestyle=:dash, label="Slope 1/2")
lines!(ax, dt_num, dt_num .^ (1.0) * UniformErrors[6, 1] / dt_num[1]^(1.0) * 0.9, color=:black, linestyle=:dash, label="Slope 1")

for (i, label) in enumerate(scheme_labels)
    lines!(ax, dt_num, UniformErrors[i, :], label=label * " ($(round(averageConvRates[i],digits=2)))")
end

fig[1, 2] = Legend(fig, ax)
fig
