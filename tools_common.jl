## numerical simulation of a stochastic Schrödinger equation
## for the paper 'Milstein-type Schemes for Hyperbolic SPDEs'
## original authors: 
##     Felix Kastner, University of Hamburg
##     Katharina Klioba, Delft University of Technology
## MIT License, Copyright (c) 2026

using LinearAlgebra: norm
using Statistics: mean


# uniform error for one path
uniform_error(approx, sol) = maximum(norm, eachslice(approx - sol, dims=2))

# estimate L²(Ω)-norm by rmse over all paths
L2norm(u) = sqrt.(mean(u .^ 2))

# compute the average numerical rate from some data points
avgRate(x, y) = mean(diff(log2.(y)) ./ diff(log2.(x)))

# smooth bump function centered at zero supported in [-c,c]
# and value 1 at zero
c = π / 2
bump(x) = (x^2 < c^2) * exp(1.0 / (x^2 - c^2)) * exp(1.0 / c^2)
periodicbump(x) = bump(rem2pi(x, RoundNearest)) # wrap around torus a.k.a. make periodic

# semigroup approximations 
R_EE(dt, M) = @. exp(1im * (-M/2+1:M/2)^2 * dt) # exponential
R_IE(dt, M) = @. (1 - 1im * (-M/2+1:M/2)^2 * dt)^(-1) # linear-implicit
R_CN(dt, M) = @. (1 + 1im * (-M/2+1:M/2)^2 * dt / 2) / (1 - 1im * (-M/2+1:M/2)^2 * dt / 2) # Crank-Nicolson