@gen function gmm(n)
    k ~ poisson(λ)
    w ~ dirichlet(δ * ones(k))
    means, vars = zeros(k), zeros(k)
    for j=1:k
        means[j] = ({:μ => j} ~ gaussian(ξ, 1/κ))
        vars[j] = ({:σ² => j} ~ inv_gamma(α, β))
    end
    for i=1:n
        z = ({:z => i} ~ categorical(w))
        {:y => i} ~ gaussian(means[z], vars[z])
    end
end

function make_constraints(ys, μs, σ²s, zs)
    arr_to_addrs(addr, arr) = [(addr => i, val) for (i, val) in enumerate(arr)]
    choicemap((:k, length(μs)),
              arr_to_addrs(:μ, μs)...,
              arr_to_addrs(:σ², σ²s)...,
              arr_to_addrs(:z, zs)...,
              arr_to_addrs(:y, ys)...)
end
