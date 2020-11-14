#############
# Dirichlet #
#############

struct DirichletTrace <: Gen.Trace
    param::Vector{Real}
    sample::Vector{Float64}
    score::Float64
end

struct Dirichlet <: Gen.GenerativeFunction{Vector{Float64}, DirichletTrace} end
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
    unconstrained_score = isempty(unconstrained_vals) ? 0 :
                          dirichlet_logpdf(unconstrained_param, unconstrained_vals)
    constrained_vals_sum = sum(Float64[constraints[i] for i=1:n if constrained_addrs[i]])
    @assert constrained_vals_sum < 1 + 1e-12
    unconstrained_vals = (1 - constrained_vals_sum) .* unconstrained_vals

    sample = [constrained_addrs[i] ? constraints[i] : popfirst!(unconstrained_vals)
              for i=1:n]
    score = dirichlet_logpdf(param, sample)
    DirichletTrace(param, sample, score), score - unconstrained_score
end

Gen.simulate(d::Dirichlet, args::Tuple) = generate(d, args, choicemap())[1]

function Gen.update(tr::DirichletTrace, args::Tuple{Vector{<:Real}},
                    argdiffs::Tuple{Union{NoChange, UnknownChange}},
                    constraints::ChoiceMap)
    old_param, new_param = get_args(tr)..., args...
    old_n, new_n = length(old_param), length(new_param)
    discard, updated = choicemap(), Dict{Int, UnknownChange}()
    new_constraints = choicemap()
    for i=1:new_n
        if has_value(constraints, i)
            new_constraints[i] = constraints[i]
        elseif has_value(get_choices(tr), i) 
            new_constraints[i] = tr[i]
        end
    end
    new_tr, generate_score = generate(dirichlet, args, new_constraints)
    for i=1:old_n
        if !has_value(get_choices(new_tr), i) || tr[i] != new_tr[i]
            discard[i] = tr[i]
        end
    end
    for i=1:max(new_n, old_n)
        if !has_value(get_choices(new_tr), i) ||
            !has_value(get_choices(tr), i) || tr[i] != new_tr[i]
            updated[i] = UnknownChange()
        end
    end

    weight = generate_score - get_score(tr)
    retdiff = isempty(updated) ? NoChange() : VectorDiff(new_n, old_n, updated)
    new_tr, weight, retdiff, discard
end
