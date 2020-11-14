@gen function split_merge_randomness(tr)
    k = tr[:k]
    prob_merge = k == 1 ? 0.0 : 0.5
    is_merge ~ bernoulli(prob_merge)
    if is_merge
        {*} ~ merge_randomness(tr)
    else
        {*} ~ split_randomness(tr)
    end
end

@gen function merge_randomness(tr)
    # r1 is the index of the first component to be merged
    # r2 + [r2 >= r1] is the index of the second component to be merged
    # [] denotes Iverson's bracket
    # j is the index of the merged component
    k = tr[:k]
    r1 ~ uniform_discrete(1, k)
    r2 ~ uniform_discrete(1, k - 1)
    j_star ~ uniform_discrete(1, k - 1)
end

@gen function split_randomness(tr)
    k, n = tr[:k], get_args(tr)...
    j_star ~ uniform_discrete(1, k)
    r1 ~ uniform_discrete(1, k + 1)
    r2 ~ uniform_discrete(1, k)
    u1 ~ beta(2, 2)
    u2 ~ beta(2, 2)
    u3 ~ beta(1, 1)

    w, σ², μ = tr[:w => j_star], tr[:σ² => j_star], tr[:μ => j_star]
    j1, j2 = r1, r2 + (r2 >= r1)
    w1, w2 = w * u1, w * (1 - u1)
    μ1, μ2 = μ - u2 * sqrt(σ² * w2/w1), μ + u2 * sqrt(σ² * w1/w2)
    σ1², σ2² = u3 * (1 - u2^2) * σ² * w/w1, (1 - u3) * (1 - u2^2) * σ² * w/w2

    for i=1:n
        y, z = tr[:y => i], tr[:z => i]
        if z == j_star
            p = w1 * exp(logpdf(gaussian, y, μ1, σ1²))
            q = w2 * exp(logpdf(gaussian, y, μ2, σ2²))
            {:to_first => i} ~ bernoulli(p/(p + q))
        end
    end
end

@transform split_merge_involution (tr_in, ch_in) to (tr_out, ch_out) begin
    is_merge = @read(ch_in[:is_merge], :disc)
    @write(ch_out[:is_merge], !is_merge, :disc)
    n = get_args(tr_in)[1]
    #for i=1:n println("$i => $(@read(tr_in[:z => i], :disc))") end
    if is_merge
        @tcall merge_transform()
    else
        @tcall split_transform()
    end
end

@transform merge_transform (tr_in, ch_in) to (tr_out, ch_out) begin
    j_star = @read(ch_in[:j_star], :disc)
    r1 = @read(ch_in[:r1], :disc)
    r2 = @read(ch_in[:r2], :disc)

    j1, j2 = r1, r2 + (r2 >= r1)
    #println("attempting to merge $j1, $j2 -> $j_star")
    w1, w2 = @read(tr_in[:w => j1], :cont), @read(tr_in[:w => j2], :cont)
    μ1, μ2 = @read(tr_in[:μ => j1], :cont), @read(tr_in[:μ => j2], :cont)
    σ1², σ2² = @read(tr_in[:σ² => j1], :cont), @read(tr_in[:σ² => j2], :cont)

    w, μ, σ² = merged_component_params(w1, μ1, σ1², w2, μ2, σ2²)
    u1, u2, u3 = reverse_split_params(w, μ, σ², w1, μ1, σ1², w2, μ2, σ2²)

    @tcall perform_merge(j1, j2, j_star, w, μ, σ²)
    @tcall specify_reverse_split(r1, r2, j_star, u1, u2, u3)
end

@transform perform_merge(j1, j2, j_star, w, μ, σ²) (tr_in, ch_in) to (tr_out, ch_out) begin
    k, n = @read(tr_in[:k], :disc), get_args(tr_in)...
    @write(tr_out[:k], k - 1, :disc)
    for i=1:n
        z = @read(tr_in[:z => i], :disc)
        if z == j1 || z == j2
            #println("in merge: $i => $j_star")
            @write(tr_out[:z => i], j_star, :disc)
            @write(ch_out[:to_first => i], z == j1, :disc)
        else
            @write(tr_out[:z => i], merge_idx(z, k, j_star, j1, j2), :disc)
        end
    end
    @write(tr_out[:w => j_star], w, :cont)
    @write(tr_out[:μ => j_star], μ, :cont)
    @write(tr_out[:σ² => j_star], σ², :cont)
    for j=1:k
        if j != j1 && j != j2
            new_idx = merge_idx(j, k, j_star, j1, j2)
            @copy(tr_in[:w => j], tr_out[:w => new_idx])
            @copy(tr_in[:μ => j], tr_out[:μ => new_idx])
            @copy(tr_in[:σ² => j], tr_out[:σ² => new_idx])
        end
    end
end

@transform specify_reverse_split(r1, r2, j_star, u1, u2, u3) (tr_in, ch_in) to (tr_out, ch_out) begin
    @write(ch_out[:j_star], j_star, :disc)
    @write(ch_out[:r1], r1, :disc)
    @write(ch_out[:r2], r2, :disc)
    @write(ch_out[:u1], u1, :cont)
    @write(ch_out[:u2], u2, :cont)
    @write(ch_out[:u3], u3, :cont)
end

@transform split_transform (tr_in, ch_in) to (tr_out, ch_out) begin
    n, = get_args(tr_in)
    k = @read(tr_in[:k], :disc)
    j_star = @read(ch_in[:j_star], :disc)
    r1 = @read(ch_in[:r1], :disc)
    r2 = @read(ch_in[:r2], :disc)
    u1 = @read(ch_in[:u1], :cont)
    u2 = @read(ch_in[:u2], :cont)
    u3 = @read(ch_in[:u3], :cont)
    
    w = @read(tr_in[:w => j_star], :cont)
    μ = @read(tr_in[:μ => j_star], :cont)
    σ² = @read(tr_in[:σ² => j_star], :cont)

    j1, j2 = r1, r2 + (r2 >= r1)
    #println("attempting to split $j_star -> $j1, $j2")
    w1, w2 = w * u1, w * (1 - u1)
    μ1, μ2 = μ - u2 * sqrt(σ² * w2/w1), μ + u2 * sqrt(σ² * w1/w2)
    σ1², σ2² = u3 * (1 - u2^2) * σ² * w/w1, (1 - u3) * (1 - u2^2) * σ² * w/w2

    @write(tr_out[:k], k + 1, :disc)
    for i=1:n
        z = @read(tr_in[:z => i], :disc)
        if z == j_star
            to_first = @read(ch_in[:to_first => i], :disc)
            #println("in split: $i => $(to_first ? j1 : j2)")
            @write(tr_out[:z => i], to_first ? j1 : j2, :disc)
        else
            @write(tr_out[:z => i], split_idx(z, k, j_star, j1 ,j2), :disc)
        end
    end
    @write(tr_out[:w => j1], w1, :cont)
    @write(tr_out[:μ => j1], μ1, :cont)
    @write(tr_out[:σ² => j1], σ1², :cont)
    @write(tr_out[:w => j2], w2, :cont)
    @write(tr_out[:μ => j2], μ2, :cont)
    @write(tr_out[:σ² => j2], σ2², :cont)
    for j=1:k
        if j != j_star
            new_idx = split_idx(j, k, j_star, j1, j2)
            @copy(tr_in[:w => j], tr_out[:w => new_idx])
            @copy(tr_in[:μ => j], tr_out[:μ => new_idx])
            @copy(tr_in[:σ² => j], tr_out[:σ² => new_idx])
        end
    end

    @copy(ch_in[:j_star], ch_out[:j_star])
    @copy(ch_in[:r1], ch_out[:r1])
    @copy(ch_in[:r2], ch_out[:r2])
end

function merged_component_params(w1, μ1, σ1², w2, μ2, σ2²)
    w = w1 + w2
    μ = (w1*μ1 + w2*μ2) / w
    σ² = -μ^2 + (w1*(μ1^2 + σ1²) + w2*(μ2^2 + σ2²)) / w
    w, μ, σ²
end

function reverse_split_params(w, μ, σ², w1, μ1, σ1², w2, μ2, σ2²)
    u1 = w1/w
    u2 = (μ - μ1) / sqrt(σ² * w2/w1)
    u3 = σ1²/σ² * u1 / (1 - u2^2)
    u1, u2, u3
end

function split_idx(j, k, j_star, j1, j2)
    j == j_star ? throw(ArgumentError("component $j will be removed")) : 0
    shift1 = -(j > j_star)
    shift2 = (j + shift1) >= min(j1, j2)
    shift3 = (j + shift1 + shift2) >= max(j1, j2)
    j + shift1 + shift2 + shift3
end

function merge_idx(j, k, j_star, j1, j2)
    j in (j1, j2) ? throw(ArgumentError("component $j will be removed")) : 0
    shift1 = -(j > min(j1, j2))
    shift2 = -(j > max(j1, j2))
    shift3 = (j + shift1 + shift2) >= j_star
    j + shift1 + shift2 + shift3
end
