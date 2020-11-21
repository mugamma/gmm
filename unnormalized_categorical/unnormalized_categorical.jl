using GenWorldModels: SetDict, item_to_indices

# TODO: use some type of vector choicemap
struct UnnormalizedCategoricalChoiceMap{ObjType} <: Gen.AddressTree{Value}
    samples::PersistentVector{ObjType}
end
function Gen.get_subtree(c::UnnormalizedCategoricalChoiceMap, i::Integer)
    if i >= 1 && i <= length(c.samples)
        Value(c.samples[i])
    else
        EmptyAddressTree()
    end
end
Gen.get_subtree(c::UnnormalizedCategoricalChoiceMap, _) = EmptyAddressTree()
function Gen.get_subtrees_shallow(c::UnnormalizedCategoricalChoiceMap)
    ((i, Value(v)) for (i, v) in enumerate(c.samples))
end

struct UnnormalizedCategoricalTrace{ObjType} <: Gen.Trace
    args::Tuple{World, Int, AbstractDict{ObjType, Real}}
    samples::PersistentVector{ObjType}
    total_weight::Float64
    obj_to_indices::SetDict
    score::Float64
end
Gen.get_gen_fn(::UnnormalizedCategoricalTrace) = unnormalized_categorical
Gen.get_args(tr::UnnormalizedCategoricalTrace) = tr.args
Gen.get_retval(tr::UnnormalizedCategoricalTrace) = tr.samples
Gen.get_score(tr::UnnormalizedCategoricalTrace) = tr.score
Gen.get_choices(tr::UnnormalizedCategoricalTrace) = UnnormalizedCategoricalChoiceMap(tr.samples)
Gen.project(::UnnormalizedCategoricalTrace, ::EmptyAddressTree) = 0.

struct UnnormalizedCategorical <: Gen.GenerativeFunction{
    PersistentVector,
    UnnormalizedCategoricalTrace
} end

"""
    list ~ unnormalized_categorical(world, num_samples, obj_to_weight)

Returns a vector with `num_samples` elements sampled with replacement from
`keys(obj_to_weight)`, where the probability that any index `i` contains
object `o` is `obj_to_weight[o] / sum(values(obj_to_weight))`.
"""
unnormalized_categorical = UnnormalizedCategorical()

function Gen.generate(
    ::UnnormalizedCategorical,
    args::Tuple{World, Int, AbstractDict{ObjType, Real}},
    constraints::ChoiceMap
) where {ObjType}
    (world, num_samples, obj_to_weight) = args
    logprob_of_sampled = 0.
    total_logprob = 0.

    pairvec = collect(obj_to_weight)
    objvec = map((x, _) -> x, pairvec)
    weightvec = map((_, y) -> y, pairvec)
    total_weight = sum(weightvec)
    weightvec /= total_weight
    samples = PersistentVector{ObjType}()
    for i=1:num_samples
        constraint = get_subtree(constraints, i)
        if isempty(constraint)
            i = categorical(weightvec)
            samples = push(samples, objvec[i])
            logprob = log(weightvec[i])
            logprob_of_sampled += logprob
            total_logprob += logprob
        else
            @assert has_value(constraint) "Constraint at index $i was a tree $constraint, not a value."
            obj = get_value(constraint)
            samples = push(samples, obj)
            total_logprob += log(obj_to_weight[obj]/totalweight)
        end
    end

    tr = UnnormalizedCategoricalTrace(args, samples, total_weight, item_to_indices(ObjType, samples), total_logprob)
    return (tr, total_logprob - logprob_of_sampled)
end

function Gen.update(
    tr::UnnormalizedCategoricalTrace,
    args::Tuple,
    (_, num_sample_diff, obj_to_weight_diff)::Tuple{GenWorldModels.WorldUpdateDiff, Diff, Union{NoChange, <:DictDiff}},
    updatespec::UpdateSpec,
    _::Selection
)
    (_, num_samples, obj_to_weight) = args
    num_changed = num_sample_diff === NoChange() ? false : get_args(tr)[1] == args[1]
    if obj_to_weight_diff === NoChange()
        obj_to_weight_diff = DictDiff(Dict(), Set(), Dict())
    end

    # we need to handle:
    # - changes in scores of unmodified samples due to the DictDiff
    # - added/removed samples due to a change in the number of samples
    # - changes to what we have sampled due to the updatespec
    
    # figure out the new total weight
    total_weight = tr.total_weight
    for (obj, weight) in obj_to_weight_diff.added
        total_weight += weight
    end
    for obj in obj_to_weight_diff.deleted
        total_weight -= get_args(tr)[3][obj]
    end
    for (obj, diff) in obj_to_new_weight_diff.updated
        if diff !== NoChange()
            total_weight -= get_args(tr)[3][obj]
            total_weight += obj_to_weight[obj]
        end
    end

    # update the samples
    objsampler = nothing
    updated = Dict{Int, <:Diff}()
    discard = choicemap()
    samples = tr.samples
    obj_to_indices = tr.obj_to_indices
    total_logprob = get_score(tr)
    total_logprob_of_chocies = 0.
    num_handled_for_obj = Dict() # TODO: accumulate in this dictionary as we go!
    total_num_handled = 0 # TODO: accumulate this
    for (i, subtree) in get_subtrees_shallow(updatespec)
        if i > num_samples
            error("Constraint provided for sample $i, but num_samples is $num_samples")
        elseif i > length(tr.samples)
            continue
        end
        if subtree isa Value
            obj = get_value(subtree)
            prob = obj_to_weight[obj]/total_weight
            total_logprob += log(prob)
        elseif subtree === AllSelection()
            if objsampler === nothing
                objsampler = get_obj_sampler(obj_to_weight)
            end
            (obj, logprob) = objsampler()
            total_logprob_of_choices += logprob
            total_logprob += logprob
        else
            error("Unrecognized UpdateSpec at index $i: $subtree")
        end
        diff[i] = UnknownChange()
        discard[i] = tr.samples[i]
        total_logprob -= get_args(tr)[3][tr.samples[i]] / tr.total_weight
        obj_to_indices = dissoc(obj_to_indices, tr.samples[i], i)
        obj_to_indices = assoc(obj_to_indices, obj, i)
        samples = assoc(samples, i, obj)
        num_handled_for_obj[obj] = get(num_handled_for_obj, obj, 0) + 1
        total_num_handled += 1
    end

    # handle additions/deletions
    if num_changed
        for i=num_samples:-1:get_args(tr)[2]
            # delete these samples
            obj = samples[i]
            samples = pop(samples)
            total_logprob -= get_args(tr)[3][tr.samples[i]] / tr.total_weight
            obj_to_indices = dissoc(obj_to_indices, tr.samples[i], i)
            discard[i] = tr.samples[i]
        end
        for i=num_samples:get_args(tr)[2]
            # generate these samples
            spec = get_subtree(updatespec, i)
            if spec isa Value
                obj = get_value(spec)
                prob = obj_to_weight[obj]/total_weight
                total_logprob += log(prob)
            elseif isempty(spec) || spec === AllSelection()
                if objsampler === nothing
                    objsampler = get_obj_sampler(obj_to_weight)
                end
                (obj, logprob) = objsampler()
                total_logprob_of_choices += logprob
                total_logprob += logprob    
            else
                error("Unrecognized UpdateSpec at index $i: $spec")
            end
            samples = push(samples, obj)
            obj_to_indices = assoc(obj_to_indices, obj, i)
            num_handled_for_obj[obj] = get(num_handled_for_obj, obj, 0) + 1
            total_num_handled += 1
        end
    end

    # handle weight changes
    log_total_ratio = log(total_weight) - log(tr.total_weight)
    for (obj, diff) in obj_to_new_weight_diff.updated
        if diff !== NoChange()
            log_weight_change = log(obj_to_weight[obj]) - log(get_args(tr)[3][obj])
            num_unhandled_occurances = length(obj_to_indices[obj]) - num_handled_for_obj[obj]
            total_logprob += num_unhandled_occurances * log_weight_change
        end
    end
    total_logprob += (num_samples - total_num_handled) * (-total_log_ratio)

    new_tr = UnnormalizedCategoricalTrace(args, samples, total_weight, obj_to_indices, total_logprob)
    diff = VectorDiff(length(tr.samples), num_samples, updated)
    weight = total_logprob - total_logprob_of_choices
    
    return (new_tr, weight, diff, discard)
end