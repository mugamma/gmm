import Distributions 

#############
# Dirichlet #
#############

struct DirichletTrace <: Gen.Trace
    param::Vector{Real}
    sample::Vector{Float64}
    score::Float64
end

struct Dirichlet <: Gen.GenerativeFunction{Vector{Real}, DirichletTrace} end
dirichlet = Dirichlet()

Gen.get_gen_fn(::DirichletTrace) = dirichlet
Gen.get_args(tr::DirichletTrace) = (tr.param,)
Gen.get_retval(tr::DirichletTrace) = tr.sample
Gen.get_score(tr::DirichletTrace) = tr.score
Gen.get_choices(tr::DirichletTrace) = choicemap(enumerate(tr.sample)...)
Gen.project(::DirichletTrace, ::Selection) = 0.

(d::Dirichlet)(param) = get_retval(simulate(d, (param,)))

function dirichlet_logpdf(param, sample) 
    if any(param .< 0) || abs(sum(sample) - 1) > 1e-12 || any(sample .< 0)
        -Inf
    else
        Distributions.logpdf(Distributions.Dirichlet(param), sample)
    end
end

function Gen.generate(::Dirichlet, (param,)::Tuple{Vector{<:Real}},
                      constraints::ChoiceMap)
    n = length(param)
    constrained_addrs = [has_value(constraints, i) for i=1:n]
    unconstrained_param = param[.!constrained_addrs]
    unconstrained_vals = rand(Distributions.Dirichlet(unconstrained_param))
    unconstrained_score = dirichlet_logpdf(unconstrained_param, unconstrained_vals)
    constrained_vals_sum = sum(Float64[constraints[i] for i=1:n if constrained_addrs[i]])
    @assert constrained_vals_sum < 1
    unconstrained_vals = (1 - constrained_vals_sum) .* unconstrained_vals

    sample = [constrained_addrs[i] ? constraints[i] : popfirst!(unconstrained_vals)
              for i=1:n]
    score = dirichlet_logpdf(param, sample)
    DirichletTrace(param, sample, score), score - unconstrained_score
end

Gen.simulate(d::Dirichlet, args::Tuple) = generate(d, args, choicemap())[1]

function Gen.update(tr::DirichletTrace, args::Tuple{Vector{<:Real}},
                    argdiffs::Tuple{NoChange}, constraints::ChoiceMap)
    param, n = get_args(tr)..., length(get_retval(tr))
    old_sample = get_retval(tr)
    new_sample = copy(old_sample)
    discard, updated = choicemap(), Dict{Int, UnknownChange}()
    for (idx, val) in get_values_shallow(constraints)
        new_sample[idx] = val
        discard[idx] = old_sample[idx]
        updated[idx] = UnknownChange()
    end

    new_score = dirichlet_logpdf(param, new_sample)
    new_tr = DirichletTrace(param, new_sample, new_score)
    weight = new_score - get_score(tr)
    retdiff = isempty(updated) ? NoChange() : VectorDiff(n, n, updated)
    new_tr, weight, retdiff, discard
end

"""
    dirichlet(alpha::AbstractVector{T}) where {T<:Real}
Sample a `Vector{Float64}` from the Dirichlet distribution with parameter
vector `alpha`.
"""
#=const dirichlet = Dirichlet()

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
Gen.has_argument_grads(::Dirichlet) = (false,)=#

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
