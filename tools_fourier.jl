## numerical simulation of a stochastic Schrödinger equation
## for the paper 'Milstein-type Schemes for Hyperbolic SPDEs'
## original authors: 
##     Felix Kastner, University of Hamburg
##     Katharina Klioba, Delft University of Technology
## MIT License, Copyright (c) 2026

using FFTW
using DSP
using CairoMakie

global kk::Vector{Float64}
global DFT_plan::AbstractFFTs.Plan{ComplexF64}
global IDFT_plan::AbstractFFTs.Plan{ComplexF64}
global conv_tmp::Vector{ComplexF64}

function setupFourierTools(m)
    global M = m
    global kk = -M/2+1:M/2
    global xx = range(0, 2π, length=M)

    u = zeros(ComplexF64, M)
    global DFT_plan = plan_ifft(u, flags=FFTW.MEASURE)
    global IDFT_plan = plan_fft(u, flags=FFTW.MEASURE)

    u2 = zeros(ComplexF64, 2M)
    global DFT_plan2 = plan_ifft(u2, flags=FFTW.MEASURE)
    global IDFT_plan2 = plan_fft(u2, flags=FFTW.MEASURE)

    global conv_tmp = zeros(ComplexF64, 2 * M - 1)
    nothing
end

# index M/2-1 corresponds to zero frequency
# this shifts the zero to the first index (=1) and back
shift(u) = circshift(u, 1 - length(u) ÷ 2)
ishift(u) = circshift(u, length(u) ÷ 2 - 1)

# discrete Fourier transform and inverse discrete Fourier transform
dft(u) = ishift(DFT_plan * u * √(2π))    # ishift(ifft(u) * √(2π))
idft(Fu) = IDFT_plan * shift(Fu) / √(2π) # fft(shift(Fu)) / √(2π)

# truncated convolution (result has the same frequencies as the inputs)
# this computes the Fourier coefficients of the product u⋅v: F(u⋅v) = 1/√2π Fu*Fv
function conv_trunc!(tmp, u, v)
    # assume length(u) == length(v) == M
    # u, v represent Fourier coeff. for frequencies -M/2+1 : M/2
    M = length(u)
    # full convolution has frequencies -M+2:M
    conv!(tmp, u, v)
    # so we only take the middle part
    # we have an extra factor to account for the chosen Fourier basis
    # where F(u⋅v) = 1/√2π Fu*Fv
    1 / √(2π) * tmp[M÷2:M+M÷2-1]
end
conv_trunc(u, v) = conv_trunc!(conv_tmp, u, v)

# plotting methods
posplot(u) = series(range(0, 2π, length=length(u)), [real.(u) imag.(u)]', axis=(xticks=([0, π, 2π], [L"0", L"\frac{\tau}{2}", L"\tau"]),))
freqplot(Fu) = series(-length(Fu)/2+1:length(Fu)/2, [real.(Fu) imag.(Fu)]', axis=(xticks=([-length(Fu) / 2 + 1, 0, length(Fu) / 2], ["$(-length(Fu)/2+1)", "0", "$(length(Fu)/2)"]),))

# given ℱ[u] compute ℱ[f(u)]
function applyFourierFunc(f, Fu)
    M = length(Fu)
    # we double the input length to reduce aliasing effects
    padFu = [zeros(M ÷ 2); Fu; zeros(M ÷ 2)]
    (DFT_plan2*f.(IDFT_plan2 * padFu))[M÷2+1:M+M÷2] # dft(f.(idft(padFu)))[M÷2+1:M+M÷2]
end
