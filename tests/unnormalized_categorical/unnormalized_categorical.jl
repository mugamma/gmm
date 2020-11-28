module TestUnnormalizedCategorical

using Test
using Gen
using GenWorldModels
using Distributions
using Profile
include("../../dirichlet.jl")

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
        @test isapprox(weight, 0.; atol=10e-10)
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

# # model equivalent to:
# # @type Cluster
# # @oupm dirichlet_cat(alpha) begin
# #     @number Cluster() ~ poisson(10)
# #     @property weight(::Cluster) ~ gamma(@arg alpha, 1)
# #     @observation_model (static, diffs) function get_clusters(num_samples)
# #         cluster_to_weight = @map (@get(weight[c]) for c in @objects(Cluster))
# #         samples ~ unnormalized_categorical(@world, num_samples, cluster_to_weight)
# #         indices = @map [@index(c) for c in samples]
# #         return indices
# #     end
# # end
@type Cluster
α = 0.9
@dist weight(::Cluster, ::World) = gamma(α, 1)
@dist num_cluster(::Tuple{}, ::World) = poisson(5)
@gen (static, diffs) function root(world, num_samples)
    clusters ~ get_sibling_set(:Cluster, :num_cluster, world, ())
    cluster_to_cluster = lazy_set_to_dict_map(identity, clusters)
    cluster_to_weight ~ dictmap_lookup_or_generate(world[:weight], cluster_to_cluster)
    samples ~ unnormalized_categorical(world, num_samples, cluster_to_weight)
    indices ~ map_lookup_or_generate(world[:index], samples)
    return indices
end
@load_generated_functions()
dc_world = UsingWorld(root, :weight => weight, :num_cluster => num_cluster)

# @gen function dc_vanilla(num_samples)
#     n ~ poisson(5)
#     probs ~ dirichlet([α for _=1:n])
#     samples ~ Map(categorical)(fill(probs, num_samples))
#     return samples
# end
@gen function un_dc_vanilla(num_samples)
    n ~ poisson(5)
    un_probs = []
    for i=1:n
        push!(un_probs, {:weight => i} ~ gamma(α, 1))
    end
    probs = un_probs/sum(un_probs)
    samples ~ Map(categorical)(fill(probs, num_samples))
    return samples
end

@testset "unnormalized categorical to emulate dirichlet->categorical" begin
    tr, weight = generate(dc_world, (10,))
    @test isapprox(weight, 0., atol=10e-10)

    num_clusters = tr[:world => :num_cluster => ()]
    ch = choicemap(
        (:n, num_clusters),
        (
            (:weight => i, tr[:world => :weight => Cluster(i)]) for i=1:num_clusters
        )...
    )
    for i=1:10
        ch[:samples => i] = get_retval(tr)[i]
    end
    van_tr, _ = generate(un_dc_vanilla, (10,), ch)

    @test isapprox(get_score(tr), get_score(van_tr))

    for _=1:10
        new_len = uniform_discrete(6, 15)
        new_num_clusters = uniform_discrete(max(1, num_clusters - 1), num_clusters + 2)
        wch = choicemap(
            (:world => :num_cluster => (), new_num_clusters)
        )
        ch = choicemap((:n, new_num_clusters))
        for i=1:min(num_clusters, new_num_clusters)
            if bernoulli(0.3)
                wch[:world => :weight => Cluster(i)] = gamma(α, 1)
                ch[:weight => i] = wch[:world => :weight => Cluster(i)]
            end
        end
        for i=num_clusters:new_num_clusters
            wch[:world => :weight => Cluster(i)] = gamma(α, 1)
            ch[:weight => i] = wch[:world => :weight => Cluster(i)]
        end
        for i=1:new_len
            if  i > 10 || get_retval(tr)[i] > new_num_clusters
                idx = uniform_discrete(1, new_num_clusters)
                wch[:kernel => :samples => i] = Cluster(idx)
                ch[:samples => i] = idx
            end
        end
        new_tr, weight, retdiff, discard = update(tr, (new_len,), (UnknownChange(),), wch)
        new_vtr, vweight, vretdiff, vdiscard = update(van_tr, (new_len,), (UnknownChange(),), ch)
        
        @test get_retval(new_tr) == get_retval(new_vtr)
        @test isapprox(weight, vweight)
        @test isapprox(get_score(new_tr), get_score(new_vtr))
    end
end

# performance testing:
# generate(dc_world, (1000,), choicemap((:world => :num_cluster => (), 10)))
# stats = @timed generate(dc_world, (100000,), choicemap((:world => :num_cluster => (), 300)))
# println("generate_time: $(stats.time)")
# tr = stats.value[1]
# for _=1:5
#     #stats = @timed
#     println("before update!")
#     stats = @timed update(tr, (100001,), (UnknownChange(),), choicemap(
#         (:world => :weight => Cluster(1), 2.)
#     ), AllSelection())
#     println("update time: $(stats.time)")
# end

end