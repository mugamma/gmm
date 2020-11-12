function mcmc_kernel(tr)
    tr, acc = mh(tr, update_w, ())
    @assert acc "update_w not gibbs"
    tr, acc = mh(tr, update_means, ())
    @assert acc "update_means not gibbs"
    tr, acc = mh(tr, update_vars, ())
    @assert acc "update_vars not gibbs"
    tr, acc = mh(tr, update_allocations, ())
    @assert acc "update_allocations not gibbs"
    # split/merge
    tr
end

@gen function update_w(tr)
    n, k = get_args(tr)..., tr[:k]
    counts = zeros(k)
    for i=1:n
        counts[tr[:z => i]] += 1
    end
    w ~ dirichlet(δ * ones(k) + counts)
end

@gen function update_means(tr)
    n, k = get_args(tr)..., tr[:k]
    for j=1:k
        y_js = [tr[:y => i] for i=1:n if tr[:z => i] == j]
        n_j, μ_j, σ²_j = length(y_js), tr[:μ => j], tr[:σ² => j]
        {:μ => j} ~ gaussian((sum(y_js)/σ²_j + κ * ξ)/(n_j/σ²_j + κ),
                             1/(n_j/σ²_j + κ))
    end
end

@gen function update_vars(tr)
    n, k = get_args(tr)..., tr[:k]
    for j=1:k
        y_js = [tr[:y => i] for i=1:n if tr[:z => i] == j]
        n_j, μ_j, σ²_j = length(y_js), tr[:μ => j], tr[:σ² => j]
        {:σ² => j} ~ inv_gamma(α + n_j/2, β + sum((y_js .- μ_j).^2)/2)
    end
end

@gen function update_allocations(tr)
    n, k, w = get_args(tr)..., tr[:k], tr[:w]
    μs = [tr[:μ => j] for j=1:k]
    σ²s = [tr[:σ² => j] for j=1:k]
    for i=1:n
        y_i = tr[:y => i]
        p = [exp(logpdf(gaussian, y_i, μ, σ²)) for (μ, σ²) in zip(μs, σ²s)] .* w
        {:z => i} ~ categorical(p ./ sum(p))
    end
end

function initial_trace(ys, k, iters=3)
    # K-means for a small number of iterations
    n = length(ys)
    μs = ys[[uniform_discrete(1, n) for _=1:k]]
    zs = [uniform_discrete(1, k) for _=1:n]
    recenter(j) = let y_js = ys[zs .== j];
                      isempty(y_js) ? ys[uniform_discrete(1, n)] : mean(y_js) end
    for _=1:iters
        μs = map(recenter, 1:k)
        zs = map(y->argmin((y .- μs).^2), ys)
    end
    σ²s = map(j->mean((ys[zs .== j] .- μs[j]).^2), 1:k)
    tr, = generate(gmm, (n,), make_constraints(ys, μs, σ²s, zs))
    tr
end
