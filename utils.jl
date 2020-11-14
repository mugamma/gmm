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

@dist positive_poisson(λ) = poisson(λ - 1) + 1

mean(v) = sum(v) / length(v)
