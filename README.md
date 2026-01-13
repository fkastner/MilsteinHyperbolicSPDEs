# Milstein-type Schemes for Hyperbolic SPDEs
*Code to reproduce the figures in the paper*

[![Code DOI](https://img.shields.io/badge/Code_DOI-10.5281/zenodo.18229440-blue.svg)](https://doi.org/10.5281/zenodo.18229440)
[![arXiv](https://img.shields.io/badge/arXiv-2512.19647-blue.svg)](https://arxiv.org/abs/2512.19647)

This is the code repository accompanying the paper "Milstein-type Schemes for Hyperbolic SPDEs".


## How to reproduce the figures

Start Julia in the root folder of this repository. Type <kbd>]</kbd> to access the package manager and activate the provided `Project.toml`:
```julia
pkg> activate .
```

Now you can start the simulations by just including the respective file
```julia-repl
julia> include("schroedinger_linear.jl")
```

This will generate a convergence plot on your screen as well as save the displayed data as a `csv` file into the `data` folder.