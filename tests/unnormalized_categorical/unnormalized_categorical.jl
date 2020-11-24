module TestUnnormalizedCategorical

using Test
using Gen
using GenWorldModels

include("../../unnormalized_categorical/unnormalized_categorical.jl")

@testset "unnormalized categorical" begin
    world = GenWorldModels.blank_world()
    num_samples = 5
    obj_to_weight = Dict(
        1 => 3,
        2 => 1
    )

    tr, wt = generate(unnormalized_categorical, (world, num_samples, obj_to_weight), choicemap((1, 1)))
    @test isapprox(wt, log(3/4))

    tr, wt = generate(unnormalized_categorical, (world, num_samples, obj_to_weight), choicemap(
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 2)
    ))
    @test isapprox(wt, log((3/4)^4 * (1/4)))
    @test get_retval(tr) == [1, 1, 1, 1, 2]

    @testset "output changing updates NoChange args" begin
        # update with a change in output
        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples, obj_to_weight), (NoChange(), NoChange(), NoChange()), choicemap((1, 2)))
        @test get_retval(new_tr) == [2, 1, 1, 1, 2]
        @test isapprox(weight, log(1/4) - log(3/4))
        @test discard == choicemap((1, 1))
        @test retdiff isa VectorDiff
        @test retdiff.prev_length == retdiff.new_length
        @test retdiff.updated == Dict(1 => UnknownChange())
        @test isapprox(get_score(new_tr) - get_score(tr), weight)

        # update with a regeneration in output
        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples, obj_to_weight), (NoChange(), NoChange(), NoChange()), select(1), EmptySelection())
        @test weight == 0.
        @test discard == choicemap((1, 1))
        @test retdiff isa VectorDiff
        @test retdiff.prev_length == retdiff.new_length
        @test retdiff.updated == Dict(1 => UnknownChange())

        # update with a regeneration, but constrained reverse move
        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples, obj_to_weight), (NoChange(), NoChange(), NoChange()), select(1), AllSelection())
        while get_retval(new_tr)[1] == 1
            (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples, obj_to_weight), (NoChange(), NoChange(), NoChange()), select(1), AllSelection())
        end
        @test isapprox(weight, -log(3/4))
        @test discard == choicemap((1, 1))
        @test retdiff isa VectorDiff
        @test retdiff.prev_length == retdiff.new_length
        @test retdiff.updated == Dict(1 => UnknownChange())

        # update, but unconstrained reverse move
        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples, obj_to_weight), (NoChange(), NoChange(), NoChange()), choicemap((1, 2)), EmptySelection())
        @test isapprox(weight, log(1/4))
        @test discard == choicemap((1, 1))
        @test retdiff isa VectorDiff
        @test retdiff.prev_length == retdiff.new_length
        @test retdiff.updated == Dict(1 => UnknownChange())
    end

    @testset "arg changing updates" begin
        # length changes:
        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples - 1, obj_to_weight), (NoChange(), UnknownChange(), NoChange()), EmptyChoiceMap())
        @test get_retval(new_tr) == [1, 1, 1, 1]
        @test isapprox(weight, -log(1/4))
        @test discard == choicemap((5, 2))
        @test retdiff isa VectorDiff
        @test retdiff.new_length == retdiff.prev_length - 1
        @test isempty(retdiff.updated)
        @test isapprox(get_score(new_tr) - get_score(tr), weight)

        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples + 1, obj_to_weight), (NoChange(), UnknownChange(), NoChange()), choicemap((6, 1)))
        @test get_retval(new_tr) == [1, 1, 1, 1, 2, 1]
        @test isapprox(weight, log(3/4))
        @test isempty(discard)
        @test retdiff isa VectorDiff
        @test retdiff.new_length == retdiff.prev_length + 1
        @test isempty(retdiff.updated)
        @test isapprox(get_score(new_tr) - get_score(tr), weight)

        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples + 1, obj_to_weight), (NoChange(), UnknownChange(), NoChange()), EmptyChoiceMap())
        @test get_retval(new_tr)[1:5] == [1, 1, 1, 1, 2]
        @test isapprox(weight, 0.; atol=10e-10)
        @test isempty(discard)
        @test retdiff isa VectorDiff
        @test retdiff.new_length == retdiff.prev_length + 1
        @test isempty(retdiff.updated)
        @test isapprox(get_score(new_tr) - get_score(tr), get_retval(new_tr)[6] == 1 ? log(3/4) : log(1/4))

        # weight changes:
        new_obj_to_weight = Dict(1 => 3, 2 => 3)
        diff = DictDiff(Dict(), Set(), Dict{Any, Diff}(2 => UnknownChange()))
        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples, new_obj_to_weight), (NoChange(), NoChange(), diff), EmptyChoiceMap())
        @test get_retval(new_tr) == [1, 1, 1, 1, 2]
        @test isapprox(get_score(new_tr), -5*log(2))
        @test isapprox(weight, get_score(new_tr) - get_score(tr))
        @test isempty(discard)
        @test retdiff isa NoChange || retdiff isa VectorDiff && retdiff.new_length == retdiff.prev_length && isempty(retdiff.updated)

        # weight change + length change:
        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples+2, new_obj_to_weight), (NoChange(), UnknownChange(), diff), EmptyChoiceMap())
        @test get_retval(new_tr)[1:5] == [1, 1, 1, 1, 2]
        @test length(get_retval(new_tr)) == 7
        @test isapprox(get_score(new_tr), -7*log(2))
        @test isapprox(weight, -5*log(2) - get_score(tr))
        @test isempty(discard)
        @test retdiff isa VectorDiff

        # weight change + length change + changed value
        new_obj_to_weight = Dict(1 => 3, 2 => 9)
        diff = DictDiff(Dict(), Set(), Dict{Any, Diff}(2 => UnknownChange()))
        (new_tr, weight, retdiff, discard) = update(tr, (world, num_samples-1, new_obj_to_weight), (NoChange(), UnknownChange(), diff), choicemap((1, 2)))
        @test get_retval(new_tr) == [2, 1, 1, 1]
        @test isapprox(get_score(new_tr), log(3/4) + 3*log(1/4))
        @test isapprox(weight, get_score(new_tr) - get_score(tr))
        @test discard == choicemap((1, 1), (5, 2))
        @test retdiff isa VectorDiff
        println("retdiff is: ", retdiff)
        @test get_args(new_tr)[2] == 4
        @test retdiff.new_length == retdiff.prev_length - 1
        @test retdiff.updated == Dict(1 => UnknownChange())
    end
end

# @gen function dirichlet_categorical(prior, num_samples)
#     probs ~ dirichlet(prior)

end