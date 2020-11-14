using Gen
import Distributions

include("utils.jl")
include("dirichlet.jl")

# Hyperparamteres
const λ = 3
const δ = 5.0
const ξ = 0.0
const κ = 0.01
const α = 2.0
const β = 10.0

include("model.jl")
include("inference.jl")
