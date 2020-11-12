import Distributions 

#############
# Dirichlet #
#############

struct Dirichlet <: Distribution{Vector{Float64}} end

"""
    dirichlet(alpha::AbstractVector{T}) where {T<:Real}
Sample a `Vector{Float64}` from the Dirichlet distribution with parameter
vector `alpha`.
"""
const dirichlet = Dirichlet()

function Gen.logpdf(::Dirichlet, x::AbstractVector{T},
                    α::AbstractVector{U}) where {T, U}
    Distributions.logpdf(Distributions.Dirichlet(α), x)
end

function Gen.logpdf_grad(::Dirichlet, x::AbstractVector{T},
                         α::AbstractVector{U}) where {T, U}
    (Distributions.gradlogpdf(Distributions.Dirichlet(alpha), x),
     nothing, nothing)
end

function Gen.random(::Dirichlet, alpha::AbstractVector{T}) where {T}
    rand(Distributions.Dirichlet(alpha))
end

(::Dirichlet)(alpha) = random(Dirichlet(), alpha)

Gen.has_output_grad(::Dirichlet) = true
Gen.has_argument_grads(::Dirichlet) = (false,)

###################
# Proper Gaussian #
###################

struct Gaussian <: Distribution{Float64} end

"""
    gaussian(mu::Real, var::Real)
Samples a `Float64` value from a normal distribution.
"""
const gaussian = Gaussian()

function Gen.logpdf(::Gaussian, x::Real, mean::Real, var::Real)
    @assert var > 0
    diff = x - mean
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

Gen.random(::Gaussian, mu::Real, var::Real) = mu + sqrt(var) * randn()
Gen.is_discrete(::Gaussian) = false

(::Gaussian)(mu, var) = random(gaussian, mu, var)

Gen.has_output_grad(::Gaussian) = false
Gen.has_argument_grads(::Gaussian) = (false, false)

##############
# Point Mass #
##############

# used for debugging 

struct PointMass <: Distribution{Real} end

const point_mass = PointMass()

Gen.logpdf(::PointMass, x::Real, point::Real) = x == point ? 0.0 : -Inf

Gen.random(::PointMass, point::Real) = point
Gen.is_discrete(::PointMass) = true

(::PointMass)(point) = random(PointMass, point)

Gen.has_output_grad(::PointMass) = false
Gen.has_argument_grads(::PointMass) = (false,)

#########
# Misc. #
#########

mean(v) = sum(v) / length(v)
