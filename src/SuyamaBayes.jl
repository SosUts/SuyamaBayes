module SuyamaBayes

using StatsFuns
using SpecialFunctions
using Distributions
using PyPlot
using ArgCheck
using BenchmarkTools

import Distributions : rand
import Flux: onehotbatch

export
    

include("pmm.jl")
# Write your package code here.

end
