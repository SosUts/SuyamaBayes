module SuyamaBayes

using StatsFuns
using SpecialFunctions
using Distributions
using PyPlot
using ArgCheck
using BenchmarkTools

import Distributions: rand

export
    # pmm.jl
    # gibbs sampling
    PriorParameters,
    PoissionMixtureModel,
    init_model,
    generate_toy_data,
    sample_s,
    sample_λ,
    sample_p,
    gibbs!,
    # variational inference
    update_s,
    update_λ,
    update_p,
    vi!

include("pmm.jl")
# Write your package code here.

end